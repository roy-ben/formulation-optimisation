# O/W NANOEMULSION - DoE SEED DESIGN with pyDOE (seed only - no modelling, no BO)
#
# Makes the starting run sheet for the MOBO campaign in Nanoemulsion-DoE-BO.py, using the SAME design factors:
#   Mixture components (% w/w, must add up to 100%):  Oil_pct, Surf_pct, Cosurf_pct, Drug_pct, Water_pct
#   Excipient identities (categorical):              Oil, Surfactant, Cosurfactant  -> 3 x 2 x 2 = 12 combinations
#
# 18 RUNS = 15 LHS runs + 3 centre-point replicates, made over 3 days (6 per day, one centre replicate each),
# all in WHOLE % w/w.


# HOW THE DESIGN IS BUILT
#   1. Latin hypercube over the 4 NON-water components. Each component's range is split into N_LHS evenly
#      spaced WHOLE-number levels (lowest and highest included), and each level is used exactly once, so every
#      component is still covered evenly from edge to edge. pyDOE's lhs() decides which levels go together.
#      Water is the "filler": Water_pct = 100 - the other four, so it's a whole number too.
#   2. Some combinations would need water outside its range. We draw many designs, throw away any with an
#      invalid recipe, and keep the best spread one that is also least correlated.
#   3. Share the 15 runs out between the 12 excipient combinations: every combination once, plus 3 extras
#      (one per oil) so each oil gets 5 runs and each surfactant / co-surfactant 7 or 8. Of many random ways of
#      doing this, keep the one where each excipient's runs are spread across the mixture space.
#   4. Add replicates of the centre recipe (rounded to whole %) to measure the noise.
#   5. Split the runs into days (RUNS_PER_DAY each). The centre replicates are spread across the days, and the
#      split is chosen so every day gets a similar mix of excipients and recipes - so a day-to-day difference
#      can't be mistaken for the effect of a factor. The run order is then randomised WITHIN each day.
#   6. Turn each recipe into mg weights for a BATCH_G batch and save the run sheet with empty measurement columns.
#
# USING IT
#   Run this file -> nanoemulsion-seed-design-pydoe.csv. Make and measure the formulations day by day in the
#   Run order, fill in Z_day0, Z_day3, Z_day7, PDI and EE_pct, save as nanoemulsion-results.csv next to
#   Nanoemulsion-DoE-BO.py and run that file - it picks up your results and suggests the first MOBO batch.
#   Z_day3 / Z_day7 are counted from the day EACH formulation was made (day 2's runs are measured a day later, etc.).
#   !! Keep the excipients and RANGES below the same as in Nanoemulsion-DoE-BO.py.
#   Most things you'd want to change are in the SETTINGS section below.
#
# A FEW PYTHON / NUMPY THINGS USED A LOT BELOW
#   numpy array   = a table of numbers. Here each ROW is one recipe and each COLUMN is one component.
#   x[:, -1]      = every row (:), last column (-1)        -> e.g. the water % of every recipe
#   x[:-1]        = everything except the last item        -> e.g. the 4 non-water components
#   x.sum(axis=1) = add up across each row                 -> e.g. the total % of each recipe
#   DataFrame     = pandas' version of a spreadsheet table, with named columns (like the CSV it gets saved as)


# =============================================================================================
# IMPORTS - the libraries this script uses
# =============================================================================================
import itertools                                         # for listing every oil/surfactant/co-surfactant combination
import math                                              # ceil / floor
import os                                                # for building file paths
import numpy as np                                       # number arrays and maths
import pandas as pd                                      # tables (DataFrames) and saving to CSV
import matplotlib.pyplot as plt                          # drawing the picture
from scipy.spatial.distance import pdist                 # distances between points (how spread out the runs are)

# [pyDOE] lhs() = Latin Hypercube Sampling. It is the ONLY pyDOE function this script uses. Its inputs:
#   n          = number of factors (columns) - here 4: oil, surfactant, co-surfactant, drug
#   samples    = number of runs (rows) - here 15
#   criterion  = 'maximin' = push the runs as far apart from each other as possible
#   iterations = how many random LHS arrangements pyDOE tries before keeping the best 'maximin' one
#   seed       = starting point for the random numbers, so the same seed always gives the same design
# It returns a table of numbers between 0 and 1. We only use their ORDER within each column (which run is
# lowest, next lowest...) to decide which whole-number level each run gets - see to_levels() below.

from pyDOE3 import lhs as _lhs                         # [pyDOE] maintained version (pip install pyDOE3)
def lhs(n, samples, criterion, iterations, seed):
    return _lhs(n, samples=samples, criterion=criterion, iterations=iterations, seed=seed)

# Output files are saved in the same folder as this script (HERE), whichever folder you run it from.
HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN_FILE = os.path.join(HERE, 'nanoemulsion-seed-design-pydoe.csv')
FIGURE_FILE = os.path.join(HERE, 'doe-seed-design-space.png')


# =============================================================================================
# SETTINGS - change these
# =============================================================================================
# --- The excipients (same as Nanoemulsion-DoE-BO.py; only the NAMES matter for the design) ---
OILS = ['Labrafac', 'Maisine', 'Squalene']
SURFACTANTS = ['Tween 80', 'Tween 20']
COSURFACTANTS = ['Span 80', 'Span 20']

# --- The mixture components (% w/w) and their ranges: (lowest, highest) ---
# !! PLACEHOLDERS - same as Nanoemulsion-DoE-BO.py. Water must be LAST: it's the filler (100 - the rest).
# Recipes use whole % only, so a range like (0.5, 10) is used as 1 - 10.
RANGES = {
  'Oil_pct':    (5, 20),
  'Surf_pct':   (2, 15),
  'Cosurf_pct': (0.5, 10),
  'Drug_pct':   (1, 10),
  'Water_pct':  (55, 90),
}
DESIGN_COLS = list(RANGES)                               # just the names: ['Oil_pct', 'Surf_pct', ..., 'Water_pct']
MEASURED_COLS = ['Z_day0', 'Z_day3', 'Z_day7', 'PDI', 'EE_pct']   # empty columns to fill in in the lab

N_LHS = 15               # LHS runs (at least 12, so every excipient combination is made at least once)
N_REPLICATES = 3         # extra repeats of the centre recipe, to measure the noise
RUNS_PER_DAY = 6         # how many formulations you make (and characterise) per day -> 18 / 6 = 3 days
N_DESIGNS = 3000         # how many LHS designs to draw and choose the best from
LHS_ITERATIONS = 20      # [pyDOE] passed to lhs() as "iterations": its own maximin search inside each draw
N_SHUFFLES = 5000        # how many random excipient assignments / day splits to try in steps 3 and 5
BATCH_G = 10             # size of each formulation (g) - the run sheet gives mg weights for this batch size
SEED = 0                 # change for a different (equally good) design. The same SEED always gives the same design


# =============================================================================================
# HELPER FUNCTIONS - small reusable pieces used further down
# =============================================================================================
def max_abs_corr(x):
  """The biggest correlation (ignoring + / -) between any two factors in the design."""
  # 0 = the two factors vary completely independently; 1 = they always go up and down together.
  corr = np.corrcoef(x, rowvar=False)
  return np.abs(corr[np.triu_indices(x.shape[1], k=1)]).max()


def to_levels(unit):
  """LHS points (0-1 scale) -> whole-% recipes. Each run gets one of the N_LHS levels of each component."""
  # argsort twice turns each column into ranks: the run with the lowest value gets 0, the next 1, ... N_LHS-1.
  # Rank k then picks level k, e.g. Oil levels 5, 6, 7, ... 20 -> the lowest-ranked run gets 5% oil.
  ranks = unit.argsort(axis=0).argsort(axis=0)
  recipes = np.column_stack([levels[:, c][ranks[:, c]] for c in range(levels.shape[1])])
  water = 100 - recipes.sum(axis=1)                      # water = whatever is left to make 100%
  return np.column_stack([recipes, water])               # stick the water column on the end


def scale01(x):
  """Every component rescaled to 0-1 (0 = its lowest allowed, 1 = its highest), so wide ranges don't dominate."""
  return (x - lows) / (highs - lows)


# Every possible (oil, surfactant, co-surfactant) trio: 3 x 2 x 2 = 12, e.g. ('Labrafac', 'Tween 80', 'Span 80').
combos = list(itertools.product(OILS, SURFACTANTS, COSURFACTANTS))
assert N_LHS >= len(combos), f'N_LHS must be at least {len(combos)} so every excipient combination gets a run'
lows = np.array([lo for lo, _ in RANGES.values()], dtype=float)    # lowest % of each component
highs = np.array([hi for _, hi in RANGES.values()], dtype=float)   # highest % of each component
assert lows.sum() <= 100 <= highs.sum(), 'These ranges can never add up to 100% - widen them'

# N_LHS evenly spaced whole-number levels for each non-water component, from its lowest to its highest whole %.
# e.g. Oil 5-20 with 15 runs -> 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20.
# When a range has fewer whole numbers than runs (e.g. Co-surfactant 1-10), some levels are used twice.
# np.floor(x + 0.5) = ordinary rounding (0.5 always goes up).
levels = np.column_stack([np.floor(np.linspace(math.ceil(lo), math.floor(hi), N_LHS) + 0.5)
                          for lo, hi in list(RANGES.values())[:-1]])


# =============================================================================================
# STEPS 1-2 - draw LHS designs, keep only the valid ones, pick the best
# =============================================================================================
# Score = spread (smallest distance between any two runs, on the 0-1 scale) x (1 - worst factor correlation).
best_recipes, best_score, n_valid = None, -np.inf, 0
for d in range(N_DESIGNS):
  # [pyDOE] This is where pyDOE actually generates the design (N_LHS rows x 4 columns, values 0-1).
  unit = lhs(len(DESIGN_COLS) - 1, samples=N_LHS, criterion='maximin', iterations=LHS_ITERATIONS, seed=SEED + d)
  recipes = to_levels(unit)
  water = recipes[:, -1]
  if np.any((water < lows[-1]) | (water > highs[-1])):  # a recipe needs too much / too little water -> reject
    continue
  n_valid += 1
  x01 = scale01(recipes)[:, :-1]                         # the 4 non-water components on the 0-1 scale
  score = pdist(x01).min() * (1 - max_abs_corr(x01))
  if score > best_score:
    best_recipes, best_score = recipes, score
assert best_recipes is not None, 'No valid LHS design found - widen the ranges or raise N_DESIGNS'
recipes = best_recipes                                   # N_LHS rows x 5 columns, all whole % w/w
recipes_01 = scale01(recipes)


# =============================================================================================
# STEP 3 - share the runs out between the excipient combinations
# =============================================================================================
# The "slots" to fill: every combination once, then the extras. Extra k goes to oil k, with a different
# (surfactant, co-surfactant) pair each time, so the oils / surfactants / co-surfactants stay balanced.
rng = np.random.default_rng(SEED)                        # random number generator (same SEED -> same result)
n_full, n_extra = divmod(N_LHS, len(combos))             # 15 runs, 12 combinations -> 1 each + 3 extras
slots = combos * n_full
pairs = list(itertools.product(SURFACTANTS, COSURFACTANTS))
pair_order = rng.permutation(len(pairs))
for k in range(n_extra):
  slots.append((OILS[k % len(OILS)],) + pairs[pair_order[k % len(pairs)]])
slots = np.array(slots, dtype=object)                    # N_LHS rows x 3 columns: oil, surfactant, co-surfactant


def imbalance(groups, x01):
  """How far the average recipe of any one group is from the average of all runs (0-1 scale; 0 = perfect).
  groups = a label for each run (e.g. its oil, or its day). A small value means no group is skewed towards
  high or low values of any component - so the effect of that group can't be mixed up with a component's."""
  overall = x01.mean(axis=0)
  return max(np.abs(x01[groups == g].mean(axis=0) - overall).max() for g in np.unique(groups))


# Try many random ways of matching recipes to slots and keep the one where:
#   - each oil's runs are spread out (smallest distance between two runs with the same oil is large), and
#   - no oil / surfactant / co-surfactant gets mostly high or mostly low values of any component.
best_assign, best_score = None, -np.inf
for _ in range(N_SHUFFLES):
  perm = rng.permutation(N_LHS)                          # recipe perm[i] goes into slot i
  x01 = recipes_01[perm]
  spread = min(pdist(x01[slots[:, 0] == oil]).min() for oil in OILS)
  worst = max(imbalance(slots[:, c], x01) for c in range(3))
  score = spread - worst
  if score > best_score:
    best_assign, best_score = perm, score

rows = []
for (oil, surf, cosurf), idx in zip(slots, best_assign):
  rows.append({'Oil': oil, 'Surfactant': surf, 'Cosurfactant': cosurf,
               'Point': 'LHS', **dict(zip(DESIGN_COLS, recipes[idx]))})


# =============================================================================================
# STEP 4 - centre-point replicates
# =============================================================================================
# The middle of each range is often not a whole number (e.g. Oil 12.5%), so try every round-down / round-up
# choice for the 4 non-water components and keep the valid one closest to the true centre (water included).
true_centre = (lows + highs) / 2
best_centre, best_dist = None, np.inf
for choice in itertools.product(*[(math.floor(m), math.ceil(m)) for m in true_centre[:-1]]):
  candidate = np.array(list(choice) + [100 - sum(choice)], dtype=float)
  if not lows[-1] <= candidate[-1] <= highs[-1]:
    continue
  dist = np.abs(scale01(candidate) - scale01(true_centre)).sum()
  if dist < best_dist:
    best_centre, best_dist = candidate, dist
assert best_centre is not None, 'No whole-% centre recipe keeps water in range'
for _ in range(N_REPLICATES):                            # made with the first excipient combination
  rows.append({'Oil': combos[0][0], 'Surfactant': combos[0][1], 'Cosurfactant': combos[0][2],
               'Point': 'Centre replicate', **dict(zip(DESIGN_COLS, best_centre))})
design = pd.DataFrame(rows)


# =============================================================================================
# STEP 5 - split into days, then randomise the order within each day
# =============================================================================================
# Day-to-day differences (room temperature, freshly made stock, instrument drift...) are a "block" effect.
#   - The centre replicates are spread across the days (with 3 days: one per day). Their spread then
#     includes the day-to-day noise - the same noise later MOBO batches will have - and comparing them across
#     days shows whether there is a day effect at all.
#   - The LHS runs are split so each day gets a similar mix of excipients AND of high / low recipes. If day 1
#     had all the high-oil runs, a day effect would look like an oil effect.
n_days = math.ceil(len(design) / RUNS_PER_DAY)
lhs_idx = np.flatnonzero(design['Point'] == 'LHS')
centre_idx = np.flatnonzero(design['Point'] == 'Centre replicate')
day = np.zeros(len(design), dtype=int)
day[centre_idx] = np.arange(len(centre_idx)) % n_days + 1          # centre replicates: day 1, 2, 1, 2...
lhs_slots_per_day = [RUNS_PER_DAY - (day[centre_idx] == d).sum() for d in range(1, n_days + 1)]
lhs_slots_per_day[-1] = len(lhs_idx) - sum(lhs_slots_per_day[:-1])   # last day takes whatever is left
lhs_day_labels = np.repeat(np.arange(1, n_days + 1), lhs_slots_per_day)
lhs_01 = scale01(design.loc[lhs_idx, DESIGN_COLS].to_numpy(dtype=float))[:, :-1]


def count_imbalance(day_labels):
  """Biggest difference between how often an excipient appears on a day and how often it 'should' (its share)."""
  worst = 0
  for col in ['Oil', 'Surfactant', 'Cosurfactant']:
    values = design.loc[lhs_idx, col].to_numpy()
    for v in np.unique(values):
      for d in range(1, n_days + 1):
        expected = (values == v).sum() * (day_labels == d).sum() / len(values)
        worst = max(worst, abs(((values == v) & (day_labels == d)).sum() - expected))
  return worst


# Keep the split with the fairest excipient counts first, then the most similar recipes between days.
best_days, best_key = None, None
for _ in range(N_SHUFFLES):
  labels = rng.permutation(lhs_day_labels)
  key = (round(count_imbalance(labels), 6), imbalance(labels, lhs_01))
  if best_key is None or key < best_key:
    best_days, best_key = labels, key
day[lhs_idx] = best_days
design['Day'] = day

# Random order within each day: shuffle the rows, then sort by day (a "stable" sort keeps the shuffled order
# inside each day). The centre replicates land wherever the shuffle puts them - no need to place them by hand.
design = design.sample(frac=1, random_state=SEED).sort_values('Day', kind='stable').reset_index(drop=True)
design.insert(0, 'Run', range(1, len(design) + 1))       # number the runs 1, 2, 3... in making order
design.insert(1, 'Day', design.pop('Day'))               # move the Day column next to Run


# =============================================================================================
# STEP 6 - convert to mg weights and save the run sheet
# =============================================================================================
# Whole % x a 10 g batch = multiples of 100 mg, so no rounding is needed; water is still "the rest" to be safe.
batch_mg = BATCH_G * 1000
MASS_COLS = [col.replace('_pct', '_mg') for col in DESIGN_COLS]   # 'Oil_pct' -> 'Oil_mg', etc.
masses = (design[DESIGN_COLS[:-1]] / 100 * batch_mg).round().astype(int)
masses.columns = MASS_COLS[:-1]
masses[MASS_COLS[-1]] = batch_mg - masses.sum(axis=1)
design[DESIGN_COLS] = design[DESIGN_COLS].astype(int)    # whole %, saved as 12 rather than 12.0
design[MASS_COLS] = masses
for column in MEASURED_COLS:                             # add empty columns for the lab results
  design[column] = np.nan
design.to_csv(DESIGN_FILE, index=False)


# --- Summary: check the design before you make anything ---
lhs_runs = design[design['Point'] == 'LHS']
lhs_runs_01 = scale01(lhs_runs[DESIGN_COLS].to_numpy(dtype=float))[:, :-1]
print(f'Seed design: {len(design)} runs ({N_LHS} LHS + {N_REPLICATES} centre replicates) over {n_days} days')
print(f'  saved to {os.path.basename(DESIGN_FILE)} (fill in: {", ".join(MEASURED_COLS)})')
print(f'  best of {n_valid} valid LHS designs (out of {N_DESIGNS} drawn)')
print(f'  smallest distance between runs (0-1 scale): {pdist(lhs_runs_01).min():.3f}')
print(f'  worst correlation between factors: {max_abs_corr(lhs_runs_01):.2f}  (aim for below ~0.3)')
print('  range covered by each component (whole %):')
for col, (lo, hi) in RANGES.items():
  print(f'    {col:11s} {lhs_runs[col].min():3d} - {lhs_runs[col].max():3d}   (allowed {lo} - {hi})')
print(f'  centre recipe: ' + ', '.join(f'{c} {v:.0f}' for c, v in zip(DESIGN_COLS, best_centre))
      + f'   ({", ".join(combos[0])})')
print('  runs per oil:', lhs_runs['Oil'].value_counts().to_dict())
print('  runs per surfactant:', lhs_runs['Surfactant'].value_counts().to_dict())
print('  runs per co-surfactant:', lhs_runs['Cosurfactant'].value_counts().to_dict())
print('  excipient combinations made twice:',
      [' / '.join(c) for c, n in lhs_runs.groupby(['Oil', 'Surfactant', 'Cosurfactant']).size().items() if n > 1])
print()
for d, runs in design.groupby('Day'):
  n_centre = (runs['Point'] == 'Centre replicate').sum()
  print(f'Day {d}: runs {runs["Run"].min()}-{runs["Run"].max()}  ({len(runs) - n_centre} LHS + {n_centre} centre)'
        f'   oils {runs.loc[runs["Point"] == "LHS", "Oil"].value_counts().to_dict()}')
  print('  mean % w/w of LHS runs: ' + ', '.join(f'{c} {runs.loc[runs["Point"] == "LHS", c].mean():.1f}'
                                                 for c in DESIGN_COLS))
print()
print(f'Weigh-out for a {BATCH_G} g batch (mg):')
print(design[['Run', 'Day', 'Point', 'Oil', 'Surfactant', 'Cosurfactant'] + MASS_COLS].to_string(index=False))


# =============================================================================================
# PICTURE - the design space, and where the seed runs sit in it
# =============================================================================================
# A "pairs plot": one panel for every pair of components (5 components -> 10 panels below the diagonal).
#   Grey cloud    = 20,000 random VALID recipes (add to 100%, every component in range) = the design space.
#   Coloured dots = the LHS seed runs (colour AND shape = oil);  black X = the centre replicates.
#   Diagonal      = how the runs spread along each component's own range.
# Whole-% recipes sit on a grid, so two runs can share a value on one axis - they still differ in the others.
cloud_unit = np.random.default_rng(SEED).random((20000, len(DESIGN_COLS) - 1))
cloud = lows[:-1] + cloud_unit * (highs[:-1] - lows[:-1])
cloud = np.column_stack([cloud, 100 - cloud.sum(axis=1)])
cloud = cloud[(cloud[:, -1] >= lows[-1]) & (cloud[:, -1] <= highs[-1])]

OIL_STYLE = {                                            # fixed colour + marker per oil, so they're never told apart by colour alone
  'Labrafac': ('#64baaa', 'o'),                          # 'o' = circle
  'Maisine':  ('#18786a', 's'),                          # 's' = square
  'Squalene': ('#0df38f', '^'),                          # '^' = triangle
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
      ax2.hist(lhs_runs[row_col], bins=10, range=RANGES[row_col], color="#154e56", alpha=0.8,
               edgecolor='white', linewidth=1.5)
      ax2.set_yticks([])
      ax2.spines[['top', 'right']].set_visible(False)
      ax.set_yticks([])
    else:
      ax.scatter(cloud[:, j], cloud[:, i], s=2, color='#b0b0b0', alpha=0.5, linewidths=0, rasterized=True)
      for oil, (colour, marker) in OIL_STYLE.items():
        runs = lhs_runs[lhs_runs['Oil'] == oil]
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
           title=f'{len(lhs_runs)} LHS runs + {len(centre_runs)} centre replicates', title_fontsize=11)
fig.suptitle('Seed design: where the runs sit in the mixture design space', fontsize=14, x=0.06, ha='left', y=0.93)
fig.savefig(FIGURE_FILE, dpi=150, bbox_inches='tight')
print(f'\nPicture saved to {os.path.basename(FIGURE_FILE)}')
