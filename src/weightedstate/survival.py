import polars as pl


def weighted_aalen_johansen(
    times: pl.Series, reals: pl.Series, weights: pl.Series
) -> pl.DataFrame:
    """Calculate a weighted Aalen-Johansen estimate.

    This function computes the probability of an event occurring over time,
    with support for weights. It can handle single events (like in Kaplan-Meier),
    or competing events (like in CIF).

    Parameters
    ----------
    times : pl.Series
        Event or censoring times.
    reals : pl.Series
        Event types: 0 for censoring, 1 for the event of interest, and 2 for
        the competing event.
    weights : pl.Series
        Non-negative observation weights.

    Returns
    -------
    pl.DataFrame
        Weighted risk-set counts, cause-specific hazard increments, overall
        survival, and cumulative incidence estimates at each observed time.
    """
    df = pl.DataFrame({"times": times, "reals": reals, "weights": weights})

    # Group by times and calculate weighted counts of events
    events_data = (
        df.group_by("times")
        .agg(
            [
                pl.when(pl.col("reals") == 0)
                .then(pl.col("weights"))
                .otherwise(0)
                .sum()
                .alias("count_0"),
                pl.when(pl.col("reals") == 1)
                .then(pl.col("weights"))
                .otherwise(0)
                .sum()
                .alias("count_1"),
                pl.when(pl.col("reals") == 2)
                .then(pl.col("weights"))
                .otherwise(0)
                .sum()
                .alias("count_2"),
            ]
        )
        .sort("times")
    )

    # Calculate total events at each time point
    events_data = events_data.with_columns(
        (pl.col("count_0") + pl.col("count_1") + pl.col("count_2")).alias(
            "events_at_times"
        )
    )

    # Calculate number of individuals at risk
    events_data = events_data.with_columns(
        pl.col("events_at_times").cum_sum(reverse=True).alias("at_risk")
    )

    # Calculate cause-specific hazards and conditional survival
    events_data = events_data.with_columns(
        [
            (pl.col("count_1") / pl.col("at_risk")).alias("csh_1"),
            (pl.col("count_2") / pl.col("at_risk")).alias("csh_2"),
        ]
    ).with_columns(
        [
            (1 - pl.col("csh_1") - pl.col("csh_2")).alias("conditional_survival"),
        ]
    )

    # Calculate overall survival
    events_data = events_data.with_columns(
        pl.col("conditional_survival").cum_prod().alias("overall_survival")
    )

    # Calculate previous overall survival
    events_data = events_data.with_columns(
        pl.col("overall_survival")
        .shift(1, fill_value=1)
        .alias("previous_overall_survival")
    )

    # Calculate transition probabilities
    events_data = events_data.with_columns(
        [
            (pl.col("csh_1") * pl.col("previous_overall_survival")).alias(
                "transition_prob_1"
            ),
            (pl.col("csh_2") * pl.col("previous_overall_survival")).alias(
                "transition_prob_2"
            ),
        ]
    )

    # Calculate state occupancy probabilities (CIF)
    events_data = events_data.with_columns(
        [
            pl.col("transition_prob_1").cum_sum().alias("cif_1"),
            pl.col("transition_prob_2").cum_sum().alias("cif_2"),
        ]
    )

    return events_data
