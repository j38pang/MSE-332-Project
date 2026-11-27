"""
prepare_outages.py -- randomly generate Provincial-grid outage events for the resilience stress test.

Each "run" is one possible 25-year future:
    number of outages in the run  ~ uniform integer between N_OUTAGES_RANGE (25-50, whole horizon)
    year of each outage           ~ uniform integer 1..25
    start hour within that year   ~ uniform over the 8760 real (chronological) hours of model_hourly.csv
    duration of each outage       ~ uniform integer between DURATION_RANGE_H (4-8 hours)
During an outage the grid connection delivers only (1 - OUTAGE_FRACTION) of its normal capacity.

Reads (processed/):
    model_hourly.csv       8760 real hours, Jan 1 -> Dec 31 (used to label each outage with its date)
    parameters.json        horizon length
Writes (processed/):
    outage_events.csv      one row per outage: run, event, year, start_t, month, day, start_hour, duration_h
    outage_params.json     outage settings as {value, unit, source} for stress_test.py

The random timings do NOT depend on OUTAGE_FRACTION: with the same SEED, a 50% and a 75% file contain the
exact same outages, so any difference in results is caused by the outage size and not by luck of the draw.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG  (change these by hand, then re-run this file before stress_test.py)
# =============================================================================
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "processed"

OUTAGE_FRACTION = 0.5          # share of grid capacity LOST during an outage: 0, 0.5, 0.75 ...
OUTAGE_BASIS = "gmax"          # "gmax":   grid limited to (1 - fraction) x G_max during the outage
N_RUNS = 200                   # number of random 25-year futures (Monte Carlo runs)
N_OUTAGES_RANGE = (25, 50)     # outages per run over the whole horizon (inclusive, uniform)
DURATION_RANGE_H = (4, 8)      # hours per outage (inclusive, uniform)
MIN_GAP_H = 24                 # at least this many normal hours between two outages, so each event is independent
SEED = 332                     # fixed seed = reproducible outages; change it for a fresh set
HOURS_PER_YEAR = 8760


# =============================================================================
# PARAMETERS
# =============================================================================
def P(value, unit, source="PLACEHOLDER"):
    return {"value": value, "unit": unit, "source": source}


PARAMS = {
    "outage_fraction":         P(OUTAGE_FRACTION, "share of grid capacity lost", "scenario setting (manual)"),
    "outage_basis":            P(OUTAGE_BASIS, "-", "scenario setting (manual)"),
    "value_of_lost_load_VoLL": P(25.0, "$/kWh",
                                 "Sullivan, Schellenberg & Blundell (2015), LBNL/Nexant, Medium & Large C&I, "
                                 "4-hr outage US$12.1/kWh (2013$) x ~1.4 inflation x ~1.38 USD->CAD, rounded"),
    "battery_soc_at_outage_start": P(0.5, "share of usable battery energy",
                                     "assumption: outages are unplanned, battery is mid-cycle when one hits"),
    "n_runs":                  P(N_RUNS, "runs", "model decision"),
    "n_outages_range":         P(list(N_OUTAGES_RANGE), "outages per 25-yr run", "project scenario"),
    "duration_range_h":        P(list(DURATION_RANGE_H), "hours", "project scenario"),
    "min_gap_h":               P(MIN_GAP_H, "hours", "model decision"),
    "seed":                    P(SEED, "-", "model decision"),
}


# =============================================================================
# RANDOM OUTAGES
# =============================================================================
def sample_run(rng, horizon):
    """One 25-year future: a list of (year, start_t, duration) with no overlaps and no event crossing a year end."""
    n = int(rng.integers(N_OUTAGES_RANGE[0], N_OUTAGES_RANGE[1] + 1))
    events, taken = [], []                      # taken = (global start, global end) of accepted events
    while len(events) < n:
        year = int(rng.integers(1, horizon + 1))
        dur = int(rng.integers(DURATION_RANGE_H[0], DURATION_RANGE_H[1] + 1))
        start = int(rng.integers(0, HOURS_PER_YEAR - dur + 1))      # whole event stays inside the year
        g0 = (year - 1) * HOURS_PER_YEAR + start                   # hour index over the full horizon
        g1 = g0 + dur
        if any(g0 < e1 + MIN_GAP_H and s0 < g1 + MIN_GAP_H for s0, e1 in taken):
            continue                                               # too close to another outage, redraw
        taken.append((g0, g1))
        events.append((year, start, dur))
    return sorted(events)


def build_events(horizon, hourly):
    rng = np.random.default_rng(SEED)
    rows = []
    for run in range(N_RUNS):
        for k, (year, start, dur) in enumerate(sample_run(rng, horizon)):
            ts = hourly.loc[start, "timestamp"]
            rows.append({"run": run, "event": k, "year": year, "start_t": start, "month": ts.month,
                         "day": ts.day, "start_hour": ts.hour, "duration_h": dur})
    return pd.DataFrame(rows)


def check(ev, horizon):
    per_run = ev.groupby("run").agg(n=("event", "size"), hours=("duration_h", "sum"))
    print(f"runs: {N_RUNS}   outage fraction: {OUTAGE_FRACTION:.0%} of capacity ({OUTAGE_BASIS} basis)")
    print(f"outages per run:  min {per_run.n.min()}  mean {per_run.n.mean():.1f}  max {per_run.n.max()}"
          f"   (target uniform {N_OUTAGES_RANGE[0]}-{N_OUTAGES_RANGE[1]}, mean {np.mean(N_OUTAGES_RANGE):.1f})")
    print(f"duration (h):     min {ev.duration_h.min()}  mean {ev.duration_h.mean():.2f}  max {ev.duration_h.max()}"
          f"   (target uniform {DURATION_RANGE_H[0]}-{DURATION_RANGE_H[1]}, mean {np.mean(DURATION_RANGE_H):.1f})")
    print(f"outage hours per run: mean {per_run.hours.mean():.0f} of {horizon * HOURS_PER_YEAR:,} "
          f"({per_run.hours.mean() / (horizon * HOURS_PER_YEAR):.3%} of the horizon)")
    print("share of outages by month:", (ev.month.value_counts(normalize=True).sort_index() * 100).round(1).to_dict())


# =============================================================================
# MAIN
# =============================================================================
def main():
    assert 0 <= OUTAGE_FRACTION <= 1 and OUTAGE_BASIS in ("gmax", "demand")
    with open(OUT_DIR / "parameters.json") as f:
        horizon = int(json.load(f)["horizon_years"]["value"])
    hourly = pd.read_csv(OUT_DIR / "model_hourly.csv", parse_dates=["timestamp"])
    assert len(hourly) == HOURS_PER_YEAR, "model_hourly.csv should have 8760 rows (run prepare_data.py first)"

    ev = build_events(horizon, hourly)
    ev.to_csv(OUT_DIR / "outage_events.csv", index=False)
    with open(OUT_DIR / "outage_params.json", "w") as f:
        json.dump(PARAMS, f, indent=2)

    print(f"Saved outage_events.csv ({len(ev)} outages) and outage_params.json to {OUT_DIR}\n")
    check(ev, horizon)


if __name__ == "__main__":
    main()
