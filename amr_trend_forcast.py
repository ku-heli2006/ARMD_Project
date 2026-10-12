"""
AMR-Specific Trend & Forecasting Module
--------------------------------------------
Tracks how resistance for specific organism-antibiotic pairs changes over
time, and produces a simple forecast for the next period.

Method: for the top N most frequent organism-antibiotic resistance pairs,
bin culture orders into monthly periods (using the jittered timestamp --
relative timing is preserved even though exact real-world dates are not),
compute the resistance rate per period, then forecast the next period
using a simple moving-average trend (a reasonable, explainable baseline;
a more advanced version could use ARIMA or Prophet).

Input:  data/armd_with_network_features.parquet
Output: prints trend summary + forecast, saves a trend chart image
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_PATH = "data/armd_with_network_features.parquet"
TOP_N_PAIRS = 5
FORECAST_WINDOW = 3  # number of recent periods to average for the forecast


def main():
    print("Loading dataset...")
    df = pd.read_parquet(INPUT_PATH)

    time_col = "order_time_jittered_utc"
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["period"] = df[time_col].dt.to_period("M")  # monthly bins

    # --- Find which AMR__ columns actually survived the earlier cleaning step ---
    amr_cols = [c for c in df.columns if c.startswith("AMR__")]
    print(f"\nAMR columns available in this dataset: {len(amr_cols)}")

    if len(amr_cols) == 0:
        print("No AMR__ columns found. Make sure you're using armd_with_network_features.parquet")
        print("(not the pre-cleaning version), and that resistance columns survived the 95%-missing cutoff.")
        return

    # --- Rank pairs by how often they actually occurred (value != -1, our "did not occur" sentinel) ---
    occurrence_counts = (df[amr_cols] != -1).sum().sort_values(ascending=False)
    top_pairs = occurrence_counts.head(TOP_N_PAIRS)
    print(f"\nTop {TOP_N_PAIRS} organism-antibiotic pairs by occurrence:")
    print(top_pairs)

    # --- For each top pair, compute resistance rate per month ---
    fig, axes = plt.subplots(len(top_pairs), 1, figsize=(10, 4 * len(top_pairs)))
    if len(top_pairs) == 1:
        axes = [axes]

    print("\n" + "=" * 70)
    print("RESISTANCE TREND + FORECAST BY ORGANISM-ANTIBIOTIC PAIR")
    print("=" * 70)

    for i, (col, count) in enumerate(top_pairs.items()):
        pair_name = col.replace("AMR__", "")
        occurred = (df[col] != -1).astype(int)

        # Resistance rate per period = occurrences in that period / total cultures in that period
        period_group = df.groupby("period").apply(
            lambda g: (g[col] != -1).sum() / len(g) if len(g) > 0 else 0
        )
        period_group = period_group.sort_index()

        if len(period_group) < 2:
            print(f"\n{pair_name}: not enough time periods to show a trend.")
            continue

        # Simple forecast: average of the last FORECAST_WINDOW periods
        recent_periods = period_group.tail(FORECAST_WINDOW)
        forecast_next = recent_periods.mean()

        # Compare early vs recent average to describe overall trend direction
        early_avg = period_group.head(len(period_group) // 3).mean()
        late_avg = period_group.tail(len(period_group) // 3).mean()
        trend_direction = "RISING" if late_avg > early_avg * 1.2 else (
            "FALLING" if late_avg < early_avg * 0.8 else "RELATIVELY FLAT"
        )

        print(f"\n{pair_name} (total occurrences: {count})")
        print(f"  Time periods covered: {len(period_group)} months "
              f"({period_group.index.min()} to {period_group.index.max()})")
        print(f"  Early-period average resistance rate: {early_avg:.2%}")
        print(f"  Recent-period average resistance rate: {late_avg:.2%}")
        print(f"  Overall trend: {trend_direction}")
        print(f"  Forecast for next period: {forecast_next:.2%}")

        # --- Plot ---
        ax = axes[i]
        period_group.plot(ax=ax, marker="o", label="Observed resistance rate")
        ax.axhline(forecast_next, color="red", linestyle="--", label=f"Forecast next period ({forecast_next:.1%})")
        ax.set_title(pair_name)
        ax.set_ylabel("Resistance rate")
        ax.legend()

    plt.tight_layout()
    plt.savefig("amr_trend_forecast.png", dpi=150)
    print(f"\nSaved trend chart to: amr_trend_forecast.png")


if __name__ == "__main__":
    main()