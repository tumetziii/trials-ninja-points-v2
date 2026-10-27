# Trials Ninja Points v2

Ninja points and Elo-style ratings for Trials Fusion ninja riders, built from leaderboard runs.

## Run it

```
pip install polars
python points_calculator.py
```

Prints the ranked leaderboard (riders with 10+ runs). `stats.py` prints per-track medians and per-level averages.

## How a run is scored

```
run points = level_worth(decimal level + fault shift) x time multiplier
```

- **Level worth:** `exp(0.223(L-1) + 0.101(L-1)^2)`. Level 1.0 = 1 point, 8.0 = 673, 9.0 = 3,824, 10.1 = 32,677. Every step is bigger than the one before; the 8-to-9 steepness is fitted to how many riders have passed each decimal level.
- **Base = an average pass.** A run at the track's median faults and median time gets exactly the level's worth.
- **Fault shift (in levels):** +1.0 for a 0f (a 0f on an 8.3 = an average pass of a 9.3), 0 at the track's median faults, down toward -0.4 for very high faults. Low faults fall off quickly: about half the 0f shift at 4% of the median.
- **Time:** +/-3% by distance from the track's median time (full 3% at two standard deviations).
- Track medians are blended with the site median (as 8 extra runs) so thin tracks don't swing.

## How a rider is rated

1. Sort runs by points; sum the best 100 with weights 1, 0.95, 0.95^2, ...
2. Effective level = the level at which 10 average passes would score the same.
3. Rating = 300 x effective level, so 3,000 means riding like a level 10. Riders need 10 runs to be ranked.

Ratings only move when a rider sets a new or better run, or when track medians or levels are recalculated.

## Data

| File | What it is |
| --- | --- |
| `data/runs_all.csv` | One row per player per track: Player, Track, Level, Faults, Time (ms) |
| `data/level_survey.tsv` | Ninja level survey votes per track (only passers vote). A track's level is the median vote, pulled toward the typical rating for its whole level as if that were 2 extra votes |
| `data/assumed_levels.tsv` | Stand-in levels for tracks not yet surveyed. Delete a row once the track has real votes |

Every tunable number is a named constant at the top of `points_calculator.py`.
