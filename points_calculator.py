
import math
import polars as pl


# CONSTANTS
# ================================================================================================================================================================================================

BASE_SCORE = 10.0                  # Base points awarded for any given run before scalers
RUN_LIMIT = 100                    # Takes the top 100 runs from a player; prevents useless stat padding
ELO_SOFT_CAP = 3000

MAX_FAULTS = 499
MAX_TIME = 30 * 60                 # Time in seconds


# Stats from pulling leaderboards - as of 10/10/26
# ================================================================================================================================================================================================

level_7_passers = 111              # Includes the true total of 81 from PC leaderboards plus a 30 player padding to adjust for anyone not on PC (mainly pre-2020 runs)
level_8_passers = 45               # Includes the true total of 42 from PC leaderboards plus a 3 player padding to adjust for anyone not on PC
level_9_passers = 8                # All on PC; true count

level_7_avg_faults = 110           # Calculated by taking the median of every level 7 track and averaging all medians
level_8_avg_faults = 110           # Calculated by taking the median of every level 8 track and averaging all medians
level_9_avg_faults = 113           # Calculated by taking the median of every level 9 track and averaging all medians

level_7_avg_time = 1187.378        # Seconds
level_8_avg_time = 1298.834        # Seconds
level_9_avg_time = 1460.378        # Seconds




# Full runs database - one row per run
# ================================================================================================================================================================================================

RUNS_SCHEMA = {
    "Player": pl.Utf8,
    "Track": pl.Utf8,
    "Level": pl.Int64,
    "Faults": pl.Int64,
    "Time": pl.Float64,
}

def load_runs(path="data/runs_all.csv"):
    return pl.read_csv(path, schema=RUNS_SCHEMA).with_columns(pl.col("Time") / 1000)  # ms -> seconds


# Run scaling based on level, faults, and time
# ================================================================================================================================================================================================


# Quadratic curve for levels 1-7, then fixed factors for 8 and 9 (no level 10 passes exist)
def level_scaler(level):
    base_rate = 0.105
    acceleration = 0.03315
    level_8_factor = 13            # About ~20 level 7 passes to outrank a single level 8 pass (rider judgement)
    level_9_factor = 20            # A typical level 9 is worth 20 level 8 passes (rider judgement, not population)

    level_7 = base_rate * 7 + acceleration * 7**2
    level_8 = level_7 * level_8_factor
    level_9 = level_8 * level_9_factor

    if level <= 7:
        return base_rate * level + acceleration * level**2
    if level == 8:
        return level_8
    return level_9


# Faults vs the level average; the curve accelerates for single digit runs, plus a 0f bonus
def fault_scaler(faults, level):
    offset = 10                    # Keeps 0f finite and sets how sharply single digits accelerate
    exponent = 0.385               # Tuned so a 0f L7 beats a median L8 pass but never an L9 pass
    zero_fault_bonus = 1.15

    avg_faults = {8: level_8_avg_faults, 9: level_9_avg_faults}.get(level, level_7_avg_faults)
    faults = min(faults, MAX_FAULTS)

    multiplier = ((avg_faults + offset) / (faults + offset)) ** exponent
    if faults == 0:
        multiplier *= zero_fault_bonus
    return multiplier


# Time vs the level average; kept weak since faults and time are correlated
def time_scaler(time, level):
    exponent = 0.15
    floor, ceiling = 0.9, 1.1

    avg_time = {8: level_8_avg_time, 9: level_9_avg_time}.get(level, level_7_avg_time)
    time = min(time, MAX_TIME)

    return min(ceiling, max(floor, (avg_time / time) ** exponent))


# Calculate each run's score
def run_score(level, faults, time):
    return BASE_SCORE * level_scaler(level) * fault_scaler(faults, level) * time_scaler(time, level)


# Scores every run, sums each player's top RUN_LIMIT runs with a decaying weight, then maps to a rating
def player_scores(runs_df):
    decay = 0.95                   # Each next-best run is worth 95% of the one before it
    rating_k = 1.48                # Small k compresses the bottom so the median player sits at ~1500
    rating_scale = 281.7           # Log curve steepness; tuned with the knee so the current top player sits at ~2880
    soft_cap_knee = 2400           # Above this rating, gains are compressed so ELO_SOFT_CAP is hard to pass
    soft_cap_width = 1925.5        # How hard the compression bites; one extra run worth 10 average L9 passes takes the top player to ~3040

    base_rating = rating_scale * (1 + pl.col("PlayerScore") / rating_k).log()
    capped_rating = soft_cap_knee + soft_cap_width * (1 + (base_rating - soft_cap_knee) / soft_cap_width).log()

    return (
        runs_df.with_columns(
            pl.struct(["Level", "Faults", "Time"])
            .map_elements(
                lambda r: run_score(r["Level"], r["Faults"], r["Time"]),
                return_dtype=pl.Float64,
            )
            .alias("RunScore")
        )
        .sort("RunScore", descending=True)
        .with_columns(pl.int_range(pl.len()).over("Player").alias("RunRank"))
        .filter(pl.col("RunRank") < RUN_LIMIT)
        .group_by("Player")
        .agg((pl.col("RunScore") * decay ** pl.col("RunRank")).sum().alias("PlayerScore"))
        .with_columns(
            pl.when(base_rating > soft_cap_knee).then(capped_rating).otherwise(base_rating)
            .round().cast(pl.Int64).alias("Rating")
        )
        .sort("PlayerScore", descending=True)
    )

runs = load_runs()
print(player_scores(runs.filter(pl.col("Player") == "PennyJX")))