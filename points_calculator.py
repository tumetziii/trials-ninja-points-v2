
import math
import statistics
import polars as pl


# CONSTANTS
# ================================================================================================================================================================================================

RUN_LIMIT = 100                    # Only a player's best 100 runs count; prevents useless stat padding
MIN_RUNS = 10                      # Runs needed to appear on the ranked leaderboard
ELO_PER_LEVEL = 300                # Rating = 300 x effective level, so 3000 means riding like a level 10

MAX_FAULTS = 499
MAX_TIME = 30 * 60                 # Time in seconds

# Level worth: worth(L) = exp(a(L-1) + b(L-1)^2 / 2), so level 1.0 = 1 point and every step is bigger than the last
LEVEL_FIRST_STEP = 1.25            # Step size at level 1.0 (rider judgement; no synced runs below level 7 yet)
LEVEL_ACCELERATION = 0.202         # How fast the step grows; fitted so 8.1 -> 9.1 matches the drop in passers (~5.8x)

# Faults move a run along the level scale instead of multiplying it down
ZERO_FAULT_SHIFT = 1.0             # A 0f counts as an average pass one full level higher (0f 8.3 = average 9.3)
WORST_FAULT_SHIFT = 0.4            # The worst possible pass counts as at most 0.4 of a level lower
LOW_FAULT_SHARPNESS = 0.04         # Smaller = low faults fall away from the 0f shift faster; 0.04 puts half the shift at ~4% of the median
MEDIAN_BLEND_RUNS = 8              # Track medians are blended with the site median as if it were this many extra runs

TIME_LIMIT = 0.03                  # Time moves a run's points by at most +/-3%
DECAY = 0.95                       # Each next-best run is worth 95% of the one before it (osu! weighting)
ELO_REFERENCE_RUNS = 10            # Effective level = the level at which this many average passes match a player's points

SURVEY_PRIOR_VOTES = 2             # Survey medians are pulled toward the typical rating for their level as if it were 2 extra votes
UNRATED_OFFSET = 0.3               # A level with no rated tracks at all is placed at whole level + 0.3


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


# Decimal ninja levels from the survey (data/level_survey.tsv), with optional stand-ins (data/assumed_levels.tsv)
# ================================================================================================================================================================================================

def load_survey(path="data/level_survey.tsv"):
    survey = pl.read_csv(path, separator="\t", schema={"Track": pl.Utf8, "Level": pl.Int64, "Scores": pl.Utf8})
    rows = []
    for track, level, scores in survey.iter_rows():
        votes = [float(v) for v in (scores or "").split()]
        rows.append({"Track": track, "Level": level, "Votes": len(votes), "Median": statistics.median(votes) if votes else None})
    return pl.DataFrame(rows, schema={"Track": pl.Utf8, "Level": pl.Int64, "Votes": pl.Int64, "Median": pl.Float64})


def level_priors(survey_df):
    """Typical rating for each whole level: the median of its rated tracks, or level + UNRATED_OFFSET if none are rated."""
    rated = survey_df.filter(pl.col("Votes") > 0)
    priors = {}
    for level in range(1, 11):
        medians = rated.filter(pl.col("Level") == level)["Median"].to_list()
        priors[level] = statistics.median(medians) if medians else level + UNRATED_OFFSET
    return priors


def round_half_up(x):
    """One decimal, halves rounded up (8.35 -> 8.4), immune to float noise like 8.349999."""
    return math.floor(x * 10 + 0.5 + 1e-9) / 10


def track_levels(runs_df, survey_path="data/level_survey.tsv", assumed_path="data/assumed_levels.tsv"):
    """Decimal level for every track: survey median shrunk toward its level's prior, then any assumed levels on top."""
    survey = load_survey(survey_path)
    priors = level_priors(survey)

    decimals = {}
    for track, level, votes, median in survey.iter_rows():
        prior = priors[level]
        decimals[track] = round_half_up((votes * median + SURVEY_PRIOR_VOTES * prior) / (votes + SURVEY_PRIOR_VOTES)) if votes else round_half_up(prior)

    if assumed_path:
        try:
            assumed = pl.read_csv(assumed_path, separator="\t")
            decimals.update(dict(zip(assumed["Track"], assumed["Level"])))
        except FileNotFoundError:
            pass

    tracks = runs_df.select("Track", "Level").unique()
    return tracks.with_columns(
        pl.struct(["Track", "Level"])
        .map_elements(lambda r: decimals.get(r["Track"], round_half_up(priors[r["Level"]])), return_dtype=pl.Float64)
        .alias("DecimalLevel")
    ).drop("Level")


# Run scaling based on level, faults, and time
# ================================================================================================================================================================================================

def level_worth(level):
    """Points for an average pass at a decimal level (polars expression). 1.0 -> 1 point, 10.1 -> ~32,700."""
    x = level - 1
    return (math.log(LEVEL_FIRST_STEP) * x + LEVEL_ACCELERATION * x * x / 2).exp()


def effective_level(points):
    """Inverse of level_worth: the level whose average pass is worth this many points."""
    a, b = math.log(LEVEL_FIRST_STEP), LEVEL_ACCELERATION
    c = points.log()
    above = 1 + (-a + (a * a + 2 * b * c.clip(lower_bound=0)).sqrt()) / b
    below = 1 + c / a
    return pl.when(c >= 0).then(above).otherwise(below).clip(lower_bound=0)


def fault_shift(faults, median):
    """Level shift from faults: +1 at 0 faults, 0 at the track median, toward -0.4 for very high faults."""
    h = LOW_FAULT_SHARPNESS * median
    at_median = 1 / (1 + median / h)
    low = ZERO_FAULT_SHIFT * (1 / (1 + faults / h) - at_median) / (1 - at_median)
    high = -WORST_FAULT_SHIFT * (faults - median) / (faults + median)
    return pl.when(faults <= median).then(low).otherwise(high)


def score_runs(runs_df, levels_df):
    """Adds DecimalLevel, FaultShift, TimeMultiplier and RunScore to every run."""
    site_median = runs_df["Faults"].median()
    log_time = pl.col("Time").clip(upper_bound=MAX_TIME).log()
    track_runs = pl.len().over("Track")

    runs = (
        runs_df.join(levels_df, on="Track", how="left")
        .with_columns(pl.col("Faults").clip(upper_bound=MAX_FAULTS))
        .with_columns(
            ((track_runs * pl.col("Faults").median().over("Track") + MEDIAN_BLEND_RUNS * site_median)
             / (track_runs + MEDIAN_BLEND_RUNS)).alias("TrackMedianFaults"),
            # Distance from the track's median time, shrunk on thin tracks (a track's only run gets 0)
            ((log_time - log_time.median().over("Track")) * (track_runs - 1) / track_runs).alias("TimeGap"),
            track_runs.alias("TrackRuns"),
        )
    )

    time_spread = runs.filter(pl.col("TrackRuns") >= 3)["TimeGap"].std()   # +/-3% is reached at 2 standard deviations

    return runs.with_columns(
        fault_shift(pl.col("Faults"), pl.col("TrackMedianFaults")).alias("FaultShift"),
        (1 + TIME_LIMIT * (-pl.col("TimeGap") / (2 * time_spread)).clip(-1, 1)).alias("TimeMultiplier"),
    ).with_columns(
        (level_worth(pl.col("DecimalLevel") + pl.col("FaultShift")) * pl.col("TimeMultiplier")).alias("RunScore")
    )


# Sums each player's top RUN_LIMIT runs with a decaying weight, then maps to a rating
# ================================================================================================================================================================================================

def player_scores(scored_df):
    totals = scored_df.group_by("Player").agg(
        pl.len().alias("Runs"),
        pl.col("DecimalLevel").max().alias("HardestPass"),
        (pl.col("Faults") == 0).sum().alias("ZeroFaults"),
    )

    points = (
        scored_df.sort("RunScore", descending=True)
        .with_columns(pl.int_range(pl.len()).over("Player").alias("RunRank"))
        .filter(pl.col("RunRank") < RUN_LIMIT)
        .group_by("Player")
        .agg((pl.col("RunScore") * DECAY ** pl.col("RunRank")).sum().alias("Points"))
    )

    return (
        points.join(totals, on="Player")
        .with_columns(effective_level(pl.col("Points") / ELO_REFERENCE_RUNS).alias("EffectiveLevel"))
        .with_columns(
            (ELO_PER_LEVEL * pl.col("EffectiveLevel")).round().cast(pl.Int64).alias("Rating"),
            (pl.col("Runs") >= MIN_RUNS).alias("Ranked"),
        )
        .sort("Points", descending=True)
    )


if __name__ == "__main__":
    runs = load_runs()
    scored = score_runs(runs, track_levels(runs))
    leaderboard = player_scores(scored)

    pl.Config.set_tbl_rows(-1)
    pl.Config.set_fmt_str_lengths(40)
    print(
        leaderboard.filter(pl.col("Ranked"))
        .with_row_index("Rank", offset=1)
        .select("Rank", "Player", "Rating", pl.col("EffectiveLevel").round(2), pl.col("Points").round(), "HardestPass", "Runs", "ZeroFaults")
    )
