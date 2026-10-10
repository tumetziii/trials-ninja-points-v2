import polars as pl


LEVELS = [7, 8, 9]


def players_per_level(runs_df):
    """Unique number of players with at least one run on each level."""
    return (
        runs_df.group_by("Level")
        .agg(pl.col("Player").n_unique().alias("Players"))
        .sort("Level")
    )


def track_medians(runs_df):
    """Median faults and time for every track."""
    return (
        runs_df.group_by("Level", "Track")
        .agg(
            pl.len().alias("Runs"),
            pl.col("Faults").median().alias("MedianFaults"),
            pl.col("Time").median().alias("MedianTime"),
        )
        .sort("Level", "Track")
    )


def level_averages(track_df):
    """Average of the track medians across all tracks of each level."""
    return (
        track_df.group_by("Level")
        .agg(
            pl.len().alias("Tracks"),
            pl.col("MedianFaults").mean().alias("AvgFaults"),
            pl.col("MedianTime").mean().alias("AvgTime"),
        )
        .sort("Level")
    )


runs = pl.read_csv("data/runs_all.csv").with_columns(pl.col("Time") / 60_000)  # ms -> minutes
players = players_per_level(runs)
tracks = track_medians(runs)
averages = level_averages(tracks)

pl.Config.set_tbl_rows(-1)
pl.Config.set_fmt_str_lengths(60)

for level in LEVELS:
    level_players = players.filter(pl.col("Level") == level)["Players"].item()
    level_tracks = tracks.filter(pl.col("Level") == level)
    avg = averages.filter(pl.col("Level") == level).row(0, named=True)

    print(f"\n===== LEVEL {level} =====")
    print(f"Unique players: {level_players}")
    print(level_tracks.drop("Level"))
    print(
        f"Average across {avg['Tracks']} tracks: "
        f"{avg['AvgFaults']:.1f} faults, {avg['AvgTime']:.2f} min"
    )
