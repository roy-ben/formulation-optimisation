# O/W NANOEMULSION - SEED DESIGN with K-MEANS CLUSTERING (seed only - no modelling, no BO)
#
# 15 + 3 runs, excipient balancing, centre replicates, 3 days of 6 - EXCEPT how the 15 recipes are picked:
#   DoE-Seed.py        -> Latin hypercube (LHS): every component's range sliced into 15 levels, each used once.
#   DoE-Seed-kmeans.py -> k-means: list every valid whole-% recipe, group them into 15 clusters of similar
#                         recipes, and make the recipe at the middle of each cluster.
#
# WHAT K-MEANS DOES HERE
#   Imagine every recipe you're allowed to make as a dot (the grey cloud in the picture). k-means splits that
#   cloud into 15 groups so that every dot is as close as possible to the middle (the "centroid") of its group.
#   Making the 15 centroids then gives 15 recipes that between them REPRESENT the whole design space: wherever
#   a recipe sits in the space, one of the 15 runs is nearby.
#   Difference from LHS: the water constraint (55-90%) is built in from the start, because only valid recipes are
#   clustered - nothing has to be thrown away. But centroids sit INSIDE their cluster, so the runs stay away
#   from the extreme edges of each range (e.g. no run at exactly 5% or 20% oil). LHS always reaches the edges.
#
# HOW THE DESIGN IS BUILT
#   1. List every valid whole-% recipe (each component in range, water = 100 - the rest, water in range).
#   2. k-means (scikit-learn) groups them into N_RUNS clusters; each centroid is moved to the nearest valid
#      whole-% recipe (no two runs the same).
#   3-6. Same as DoE-Seed.py: excipients, centre replicates, days, mg weights.
#   At the end the design is compared with the LHS one in nanoemulsion-seed-design-pydoe.csv (if it exists).
#
# USING IT
#   Run this file -> nanoemulsion-seed-design-kmeans.csv + doe-seed-design-space-kmeans.png.
#   !! Keep the excipients and RANGES below the same as in DoE-Seed.py and Nanoemulsion-DoE-BO.py.


# =============================================================================================
# IMPORTS - the libraries this script uses
# =============================================================================================
import itertools                                         # for listing every combination
import math                                              # ceil / floor
import os                                                # for building file paths
import numpy as np                                       # number arrays and maths
import pandas as pd                                      # tables (DataFrames) and saving to CSV
import matplotlib.pyplot as plt                          # drawing the picture
from scipy.spatial.distance import cdist, pdist          # distances between points
from sklearn.cluster import KMeans                       # [k-means] the clustering (pip install scikit-learn)

# Output files are saved in the same folder as this script (HERE), whichever folder you run it from.
HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN_FILE = os.path.join(HERE, 'nanoemulsion-seed-design-kmeans.csv')
FIGURE_FILE = os.path.join(HERE, 'doe-seed-design-space-kmeans.png')
LHS_DESIGN_FILE = os.path.join(HERE, 'nanoemulsion-seed-design-pydoe.csv')   # made by DoE-Seed.py, for comparing


# =============================================================================================
# SETTINGS - change these (keep them the same as DoE-Seed.py for a fair comparison)
# =============================================================================================
OILS = ['Labrafac', 'Maisine', 'Squalene']
SURFACTANTS = ['Tween 80', 'Tween 20']
COSURFACTANTS = ['Span 80', 'Span 20']

# Recipes use whole % only, so a range like (0.5, 10) is used as 1 - 10. Water must be LAST (the filler).
RANGES = {
  'Oil_pct':    (5, 20),
  'Surf_pct':   (2, 15),
  'Cosurf_pct': (0.5, 10),
  'Drug_pct':   (1, 10),
  'Water_pct':  (55, 90),
}
DESIGN_COLS = list(RANGES)
MEASURED_COLS = ['Z_day0', 'Z_day3', 'Z_day7', 'PDI', 'EE_pct']

N_RUNS = 15              # k-means runs = number of clusters (at least 12, so every excipient combination gets one)
N_REPLICATES = 3         # extra repeats of the centre recipe, to measure the noise
RUNS_PER_DAY = 6         # how many formulations you make (and characterise) per day -> 18 / 6 = 3 days
N_INIT = 50              # [k-means] how many times k-means starts again from different random centroids (keeps the best)
N_SHUFFLES = 5000        # how many random excipient assignments / day splits to try in steps 3 and 5
BATCH_G = 10             # size of each formulation (g)
SEED = 0                 # the same SEED always gives the same design


# =============================================================================================
# HELPER FUNCTIONS
# =============================================================================================
def max_abs_corr(x):
  """The biggest correlation (ignoring + / -) between any two factors in the design."""
  corr = np.corrcoef(x, rowvar=False)
  return np.abs(corr[np.triu_indices(x.shape[1], k=1)]).max()


def scale01(x):
  """Every component rescaled to 0-1 (0 = its lowest allowed, 1 = its highest), so wide ranges don't dominate."""
  return (x - lows) / (highs - lows)


def imbalance(groups, x01):
  """How far the average recipe of any one group is from the average of all runs (0-1 scale; 0 = perfect)."""
  overall = x01.mean(axis=0)
  return max(np.abs(x01[groups == g].mean(axis=0) - overall).max() for g in np.unique(groups))


combos = list(itertools.product(OILS, SURFACTANTS, COSURFACTANTS))
assert N_RUNS >= len(combos), f'N_RUNS must be at least {len(combos)} so every excipient combination gets a run'
lows = np.array([lo for lo, _ in RANGES.values()], dtype=float)
highs = np.array([hi for _, hi in RANGES.values()], dtype=float)
assert lows.sum() <= 100 <= highs.sum(), 'These ranges can never add up to 100% - widen them'


# =============================================================================================
# STEP 1 - list every valid whole-% recipe (the "candidates")
# =============================================================================================
# Every whole % for each non-water component (e.g. Oil 5, 6, ... 20), every combination of them, water fills
# to 100%, and only recipes with water in range are kept. This IS the design space, as a list of dots.
whole_levels = [range(math.ceil(lo), math.floor(hi) + 1) for lo, hi in list(RANGES.values())[:-1]]
candidates = np.array(list(itertools.product(*whole_levels)), dtype=float)
candidates = np.column_stack([candidates, 100 - candidates.sum(axis=1)])
candidates = candidates[(candidates[:, -1] >= lows[-1]) & (candidates[:, -1] <= highs[-1])]
cand_01 = scale01(candidates)[:, :-1]                    # the 4 non-water components on the 0-1 scale


# =============================================================================================
# STEP 2 - k-means: group the candidates into N_RUNS clusters, one run per cluster
# =============================================================================================
# [k-means] KMeans settings:
#   n_clusters = how many groups (= how many runs)
#   n_init     = run the whole thing N_INIT times from different random starting centroids and keep the best
#                (the one where the recipes are closest to their own centroid - "inertia")
#   init       = 'k-means++' spreads the starting centroids out, which gives better, more consistent clusters
# Clustering is done on the 0-1 scale so every component counts equally (same as the distances in DoE-Seed.py).
kmeans = KMeans(n_clusters=N_RUNS, n_init=N_INIT, init='k-means++', random_state=SEED).fit(cand_01)
centroids = kmeans.cluster_centers_                      # N_RUNS x 4, on the 0-1 scale - usually NOT whole %

# Move each centroid to the closest valid whole-% recipe. Centroids are done closest-match-first, and a recipe
# already taken can't be used again, so no two runs are identical.
dist = cdist(centroids, cand_01)
order = np.argsort(dist.min(axis=1))
taken = set()
picked = np.empty(N_RUNS, dtype=int)
for c in order:
  for idx in np.argsort(dist[c]):
    if idx not in taken:
      picked[c] = idx
      taken.add(idx)
      break
recipes = candidates[picked]                             # N_RUNS rows x 5 columns, all whole % w/w
recipes_01 = scale01(recipes)


# =============================================================================================
# STEP 3 - share the runs out between the excipient combinations (same as DoE-Seed.py)
# =============================================================================================
rng = np.random.default_rng(SEED)
n_full, n_extra = divmod(N_RUNS, len(combos))
slots = combos * n_full
pairs = list(itertools.product(SURFACTANTS, COSURFACTANTS))
pair_order = rng.permutation(len(pairs))
for k in range(n_extra):
  slots.append((OILS[k % len(OILS)],) + pairs[pair_order[k % len(pairs)]])
slots = np.array(slots, dtype=object)

best_assign, best_score = None, -np.inf
for _ in range(N_SHUFFLES):
  perm = rng.permutation(N_RUNS)
  x01 = recipes_01[perm]
  spread = min(pdist(x01[slots[:, 0] == oil]).min() for oil in OILS)
  worst = max(imbalance(slots[:, c], x01) for c in range(3))
  score = spread - worst
  if score > best_score:
    best_assign, best_score = perm, score

rows = []
for (oil, surf, cosurf), idx in zip(slots, best_assign):
  rows.append({'Oil': oil, 'Surfactant': surf, 'Cosurfactant': cosurf,
               'Point': 'k-means', **dict(zip(DESIGN_COLS, recipes[idx]))})


# =============================================================================================
# STEP 4 - centre-point replicates (same as DoE-Seed.py)
# =============================================================================================
true_centre = (lows + highs) / 2
best_centre, best_dist = None, np.inf
for choice in itertools.product(*[(math.floor(m), math.ceil(m)) for m in true_centre[:-1]]):
  candidate = np.array(list(choice) + [100 - sum(choice)], dtype=float)
  if not lows[-1] <= candidate[-1] <= highs[-1]:
    continue
  d = np.abs(scale01(candidate) - scale01(true_centre)).sum()
  if d < best_dist:
    best_centre, best_dist = candidate, d
assert best_centre is not None, 'No whole-% centre recipe keeps water in range'
for _ in range(N_REPLICATES):
  rows.append({'Oil': combos[0][0], 'Surfactant': combos[0][1], 'Cosurfactant': combos[0][2],
               'Point': 'Centre replicate', **dict(zip(DESIGN_COLS, best_centre))})
design = pd.DataFrame(rows)


# =============================================================================================
# STEP 5 - split into days, then randomise the order within each day (same as DoE-Seed.py)
# =============================================================================================
n_days = math.ceil(len(design) / RUNS_PER_DAY)
run_idx = np.flatnonzero(design['Point'] == 'k-means')
centre_idx = np.flatnonzero(design['Point'] == 'Centre replicate')
day = np.zeros(len(design), dtype=int)
day[centre_idx] = np.arange(len(centre_idx)) % n_days + 1
run_slots_per_day = [RUNS_PER_DAY - (day[centre_idx] == d).sum() for d in range(1, n_days + 1)]
run_slots_per_day[-1] = len(run_idx) - sum(run_slots_per_day[:-1])
run_day_labels = np.repeat(np.arange(1, n_days + 1), run_slots_per_day)
runs_01 = scale01(design.loc[run_idx, DESIGN_COLS].to_numpy(dtype=float))[:, :-1]


def count_imbalance(day_labels):
  """Biggest difference between how often an excipient appears on a day and how often it 'should' (its share)."""
  worst = 0
  for col in ['Oil', 'Surfactant', 'Cosurfactant']:
    values = design.loc[run_idx, col].to_numpy()
    for v in np.unique(values):
      for d in range(1, n_days + 1):
        expected = (values == v).sum() * (day_labels == d).sum() / len(values)
        worst = max(worst, abs(((values == v) & (day_labels == d)).sum() - expected))
  return worst


best_days, best_key = None, None
for _ in range(N_SHUFFLES):
  labels = rng.permutation(run_day_labels)
  key = (round(count_imbalance(labels), 6), imbalance(labels, runs_01))
  if best_key is None or key < best_key:
    best_days, best_key = labels, key
day[run_idx] = best_days
design['Day'] = day

design = design.sample(frac=1, random_state=SEED).sort_values('Day', kind='stable').reset_index(drop=True)
design.insert(0, 'Run', range(1, len(design) + 1))
design.insert(1, 'Day', design.pop('Day'))


# =============================================================================================
# STEP 6 - convert to mg weights and save the run sheet (same as DoE-Seed.py)
# =============================================================================================
batch_mg = BATCH_G * 1000
MASS_COLS = [col.replace('_pct', '_mg') for col in DESIGN_COLS]
masses = (design[DESIGN_COLS[:-1]] / 100 * batch_mg).round().astype(int)
masses.columns = MASS_COLS[:-1]
masses[MASS_COLS[-1]] = batch_mg - masses.sum(axis=1)
design[DESIGN_COLS] = design[DESIGN_COLS].astype(int)
design[MASS_COLS] = masses
for column in MEASURED_COLS:
  design[column] = np.nan
design.to_csv(DESIGN_FILE, index=False)


# --- Summary ---
km_runs = design[design['Point'] == 'k-means']
print(f'k-means seed design: {len(design)} runs ({N_RUNS} k-means + {N_REPLICATES} centre replicates) over {n_days} days')
print(f'  saved to {os.path.basename(DESIGN_FILE)} (fill in: {", ".join(MEASURED_COLS)})')
print(f'  clustered {len(candidates):,} valid whole-% recipes into {N_RUNS} groups (best of {N_INIT} k-means starts)')
print(f'  centre recipe: ' + ', '.join(f'{c} {v:.0f}' for c, v in zip(DESIGN_COLS, best_centre))
      + f'   ({", ".join(combos[0])})')
print('  runs per oil:', km_runs['Oil'].value_counts().to_dict())
print('  excipient combinations made twice:',
      [' / '.join(c) for c, n in km_runs.groupby(['Oil', 'Surfactant', 'Cosurfactant']).size().items() if n > 1])
for d, runs in design.groupby('Day'):
  n_centre = (runs['Point'] == 'Centre replicate').sum()
  print(f'  Day {d}: runs {runs["Run"].min()}-{runs["Run"].max()}  ({len(runs) - n_centre} k-means + {n_centre} centre)'
        f'   oils {runs.loc[runs["Point"] != "Centre replicate", "Oil"].value_counts().to_dict()}')


# =============================================================================================
# COMPARISON with the LHS design (DoE-Seed.py)
# =============================================================================================
# All on the 0-1 scale of the 4 non-water components. "Valid recipe" = any of the candidates from step 1.
#   Closest two runs     - bigger = no two runs are wasted on almost the same recipe
#   Worst correlation    - smaller = the model can tell the components' effects apart
#   Biggest gap          - the valid recipe FURTHEST from any run. Smaller = no large unexplored hole
#   Average gap          - how far a typical valid recipe is from its nearest run. Smaller = more representative
#   Range covered        - how much of each component's range the runs reach
def design_metrics(x):
  x01 = scale01(x)[:, :-1]
  gaps = cdist(cand_01, x01).min(axis=1)                 # each valid recipe -> distance to its nearest run
  return {
    'Closest two runs': f'{pdist(x01).min():.3f}',
    'Worst correlation': f'{max_abs_corr(x01):.2f}',
    'Biggest gap': f'{gaps.max():.3f}',
    'Average gap': f'{gaps.mean():.3f}',
    **{f'{col} range': f'{x[:, i].min():.0f}-{x[:, i].max():.0f}' for i, col in enumerate(DESIGN_COLS)},
  }


table = {'k-means': design_metrics(km_runs[DESIGN_COLS].to_numpy(dtype=float))}
lhs_runs = None
if os.path.exists(LHS_DESIGN_FILE):
  lhs_runs = pd.read_csv(LHS_DESIGN_FILE)
  lhs_runs = lhs_runs[lhs_runs['Point'] == 'LHS']
  table['LHS (DoE-Seed.py)'] = design_metrics(lhs_runs[DESIGN_COLS].to_numpy(dtype=float))
print()
print('Comparison (the 15 design runs only, centre replicates left out):')
print(pd.DataFrame(table).to_string())
if lhs_runs is None:
  print(f'  (run DoE-Seed.py first to add the LHS design to this comparison)')


# =============================================================================================
# PICTURE - same layout as DoE-Seed.py
# =============================================================================================
cloud_unit = np.random.default_rng(SEED).random((20000, len(DESIGN_COLS) - 1))
cloud = lows[:-1] + cloud_unit * (highs[:-1] - lows[:-1])
cloud = np.column_stack([cloud, 100 - cloud.sum(axis=1)])
cloud = cloud[(cloud[:, -1] >= lows[-1]) & (cloud[:, -1] <= highs[-1])]

OIL_STYLE = {
  'Labrafac': ('#64baaa', 'o'),
  'Maisine':  ('#18786a', 's'),
  'Squalene': ('#0df38f', '^'),
}

labels = {'Oil_pct': 'Oil % w/w', 'Surf_pct': 'Surfactant % w/w', 'Cosurf_pct': 'Co-surfactant % w/w',
          'Drug_pct': 'Drug % w/w', 'Water_pct': 'Water % w/w'}
centre_runs = design[design['Point'] == 'Centre replicate']
n = len(DESIGN_COLS)
fig, axes = plt.subplots(n, n, figsize=(13, 13))
for i, row_col in enumerate(DESIGN_COLS):
  for j, col_col in enumerate(DESIGN_COLS):
    ax = axes[i, j]
    ax.spines[['top', 'right']].set_visible(False)
    if j > i:
      ax.axis('off')
      continue
    if i == j:
      ax.hist(cloud[:, i], bins=24, range=RANGES[row_col], color="#b7d6d6", density=True)
      ax2 = ax.twinx()
      ax2.hist(km_runs[row_col], bins=10, range=RANGES[row_col], color="#154e56", alpha=0.8,
               edgecolor='white', linewidth=1.5)
      ax2.set_yticks([])
      ax2.spines[['top', 'right']].set_visible(False)
      ax.set_yticks([])
    else:
      ax.scatter(cloud[:, j], cloud[:, i], s=2, color='#b0b0b0', alpha=0.5, linewidths=0, rasterized=True)
      for oil, (colour, marker) in OIL_STYLE.items():
        runs = km_runs[km_runs['Oil'] == oil]
        ax.scatter(runs[col_col], runs[row_col], s=40, color=colour, marker=marker,
                   edgecolors='white', linewidths=0.8, label=oil, zorder=3)
      ax.scatter(centre_runs[col_col], centre_runs[row_col], s=60, color='black', marker='X',
                 edgecolors='white', linewidths=0.8, label='Centre replicates', zorder=4)
      ax.set_ylim(RANGES[row_col])
    ax.set_xlim(RANGES[col_col])
    ax.tick_params(labelsize=8, colors='#555555')
    if i == n - 1:
      ax.set_xlabel(labels[col_col], fontsize=9)
    else:
      ax.set_xticklabels([])
    if j == 0 and i > 0:
      ax.set_ylabel(labels[row_col], fontsize=9)
    elif j > 0:
      ax.set_yticklabels([])

handles, legend_labels = axes[1, 0].get_legend_handles_labels()
handles.append(plt.Rectangle((0, 0), 1, 1, color='#b0b0b0'))
legend_labels.append('All valid recipes (design space)')
fig.legend(handles, legend_labels, loc='upper right', bbox_to_anchor=(0.92, 0.88), frameon=False, fontsize=11,
           title=f'{len(km_runs)} k-means runs + {len(centre_runs)} centre replicates', title_fontsize=11)
fig.suptitle('k-means seed design: where the runs sit in the mixture design space', fontsize=14, x=0.06,
             ha='left', y=0.93)
fig.savefig(FIGURE_FILE, dpi=150, bbox_inches='tight')
print(f'\nPicture saved to {os.path.basename(FIGURE_FILE)}')
