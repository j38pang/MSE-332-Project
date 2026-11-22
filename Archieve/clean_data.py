"""
MSE 332 - Group 3: clean and align the UW demand and EV3 solar PV data
=======================================================================
Output: one hourly table (8760 rows) that the optimization model can read directly.

Model conventions this script produces
  * time step      : 1 hour, "hour-ending" (hour 1 = 00:00-01:00), 365 days, no Feb 29
  * demand_MW      : mean of the four 15-min kW readings inside each hour, in MW
                     (average power in MW for 1 h = energy in MWh)
  * demand_peak_MW : highest 15-min reading inside the hour (kept for peak-charge checks)
  * cf_s1..cf_s3   : PV capacity factor as a FRACTION (0-1) for the three PV scenarios
  * season         : Winter Dec-Feb, Spring Mar-May, Summer Jun-Aug, Fall Sep-Nov
  * calendar order : Jan 1 -> Dec 31 (demand is Oct 2025-Sep 2026, so it is re-ordered by
                     month/day; the only jump is Sep 30 -> Oct 1)
Usage: python3 clean_data.py <Solar_PV_data.xlsx> <DemandProfile_UW_.xlsx> <output.xlsx>
"""
import sys
import numpy as np
import pandas as pd

pv_path, dem_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
log = []  # every change we make is recorded here and written to the "Cleaning_log" sheet


def note(step, detail):
    log.append({"step": step, "detail": detail})


SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
          6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}

# ----------------------------------------------------------------------------------------
# 1. DEMAND: 15-min kW  ->  hourly MW
# ----------------------------------------------------------------------------------------
d = pd.read_excel(dem_path, header=1)          # row 0 is a title line, row 1 holds the headers
d.columns = ["date", "time", "kw", "kvar", "kva", "pf"]
d["date"] = pd.to_datetime(d["date"])

# Excel stores the 24:00 reading of each day as a "1 day" timedelta instead of a clock time,
# so convert every entry to a timedelta before adding it to the date.
def to_td(t):
    if hasattr(t, "hour"):                      # normal clock time, e.g. 01:15:00
        return pd.Timedelta(hours=t.hour, minutes=t.minute)
    return pd.Timedelta(t)                      # the 24:00 readings (datetime.timedelta of 1 day)

d["ts"] = d["date"] + d["time"].map(to_td)     # 15-min INTERVAL-ENDING timestamps
assert len(d) == 35040 and not d["ts"].duplicated().any(), "unexpected demand structure"
assert (d["ts"].diff().dropna() == pd.Timedelta(minutes=15)).all(), "gaps in 15-min series"
note("Demand: structure", "35,040 rows = 365 days x 96, no gaps, no duplicates, no blanks.")

# Interval-ending stamps: the four rows ending :15, :30, :45, :00 make up one hour-ending slot.
shifted = d["ts"] - pd.Timedelta(minutes=1)
d["month"], d["day"] = shifted.dt.month, shifted.dt.day
d["hour"] = shifted.dt.hour + 1                # hour-ending 1..24, same convention as the PV file

hourly = (d.groupby(["month", "day", "hour"])
            .agg(demand_kw=("kw", "mean"), demand_peak_kw=("kw", "max"),
                 kva_max=("kva", "max"), n=("kw", "size"))
            .reset_index())
assert (hourly["n"] == 4).all() and len(hourly) == 8760
hourly["demand_MW"] = hourly["demand_kw"] / 1000
hourly["demand_peak_MW"] = hourly["demand_peak_kw"] / 1000
note("Demand: hourly", "Hourly demand = mean of the 4 fifteen-minute kW readings per hour (hour-ending), "
     "divided by 1000 for MW. Highest 15-min value in each hour kept as demand_peak_MW.")

# Unusual values are flagged but NOT changed (they may be real events). Ask your TA if unsure.
roll = d["kw"].rolling(9, center=True, min_periods=3).median()
odd = d[(d["kw"] - roll).abs() / roll > 0.25]
for _, r in odd.iterrows():
    note("Demand: flagged, kept", f"{r['ts']}  {r['kw']:.0f} kW is >25% from its local median.")
low = d.nsmallest(1, "kw").iloc[0]
note("Demand: flagged, kept", f"Lowest reading {low['kw']:.0f} kW at {low['ts']} (typical night load is ~9,000 kW). "
     "Looks like a real low-load event on Jun 11; check before relying on the minimum.")

# ----------------------------------------------------------------------------------------
# 2. SOLAR PV: capacity factor % -> fraction, fix bad / missing data
# ----------------------------------------------------------------------------------------
p = pd.read_excel(pv_path)
p.columns = ["month", "day", "hour", "cf_s1", "cf_s2", "cf_s3"]
assert len(p) == 8760 and not p.duplicated(["month", "day", "hour"]).any()
scen = ["cf_s1", "cf_s2", "cf_s3"]

# 2a. Night mask: an hour of the day is "night" in a month if at least TWO of the three scenarios
#     are zero on every day of that month at that hour. (Using one scenario alone would wrongly
#     zero real dawn/dusk output in the others; requiring two still catches a single faulty series.)
zero_votes = sum((p.groupby(["month", "hour"])[c].transform("max").fillna(0) == 0).astype(int) for c in scen)
night = zero_votes >= 2

# 2b. Days that must be rebuilt: missing values, or a block of non-zero output at night (sensor fault)
p["doy"] = p.groupby(["month", "day"]).ngroup()
bad_days = {}
for c in scen:
    miss = set(p.loc[p[c].isna(), "doy"])
    nz_night = p[night & (p[c] > 0)].groupby("doy").size()
    faulty = set(nz_night[nz_night >= 3].index)          # 3+ night values on one day = fault, not rounding
    bad_days[c] = sorted(miss | faulty)
    stray = nz_night[nz_night < 3].sum()
    for dd in sorted(miss):
        r = p[p.doy == dd].iloc[0]
        note(f"PV {c}: missing day", f"month {int(r.month)} day {int(r.day)}: all 24 values blank -> rebuilt.")
    for dd in sorted(faulty - miss):
        r = p[p.doy == dd].iloc[0]
        note(f"PV {c}: faulty day", f"month {int(r.month)} day {int(r.day)}: non-zero output at night -> rebuilt.")
    if stray:
        keep = ~p["doy"].isin(bad_days[c])
        biggest = p.loc[night & (p[c] > 0) & keep, c].max()
        note(f"PV {c}: stray night values",
             f"{int(stray)} isolated non-zero values in hours that are dark in the other scenarios "
             f"(largest {biggest:.0f}%) set to 0.")

# 2c. Rebuild flagged days: same hour averaged over the neighbouring days (+/-5), flagged days excluded
for c in scen:
    grid = p.pivot(index="doy", columns="hour", values=c).sort_index()     # 365 x 24
    grid.loc[bad_days[c]] = np.nan
    filled = grid.copy()
    for dd in bad_days[c]:
        lo, hi = max(0, dd - 5), min(364, dd + 5)
        filled.loc[dd] = grid.loc[lo:hi].mean(skipna=True)
    p[c] = filled.stack().reindex(pd.MultiIndex.from_arrays([p["doy"], p["hour"]])).values

# 2d. Night values to zero, clip to 0-100 %, convert to a fraction
for c in scen:
    p.loc[night, c] = 0.0
    over = int((p[c] > 100).sum())
    if over:
        note(f"PV {c}: above 100%", f"{over} values above 100% clipped to 100%.")
    p[c] = p[c].clip(0, 100) / 100.0
assert p[scen].notna().all().all()

# ----------------------------------------------------------------------------------------
# 3. MERGE on (month, day, hour), add model index and season
# ----------------------------------------------------------------------------------------
m = (p[["month", "day", "hour"] + scen]
     .merge(hourly[["month", "day", "hour", "demand_MW", "demand_peak_MW", "kva_max"]],
            on=["month", "day", "hour"], validate="1:1"))
m = m.sort_values(["month", "day", "hour"]).reset_index(drop=True)
m.insert(0, "t", np.arange(1, len(m) + 1))
m["season"] = m["month"].map(SEASON)
m["CF_t"] = m[scen].mean(axis=1)               # model CF_t: average of the 3 scenarios in each hour
note("CF_t", "CF_t for each hour = average of cf_s1, cf_s2, cf_s3 in that hour (after cleaning).")
note("Merge", "Joined on month/day/hour (both files are 365 days, hour-ending). "
     "Demand year Oct 2025-Sep 2026 is placed on a Jan-Dec calendar; the Sep 30 -> Oct 1 step is the only discontinuity.")
assert len(m) == 8760 and m.notna().all().all()

# ----------------------------------------------------------------------------------------
# 4. Summaries for the report / parameters
# ----------------------------------------------------------------------------------------
season_order = ["Winter", "Spring", "Summer", "Fall"]
seas = (m.groupby("season")
          .agg(hours=("t", "size"), mean_demand_MW=("demand_MW", "mean"), peak_demand_MW=("demand_MW", "max"),
               cf_s1=("cf_s1", "mean"), cf_s2=("cf_s2", "mean"), cf_s3=("cf_s3", "mean"))
          .reindex(season_order).reset_index())
seas["energy_GWh"] = seas["hours"] * seas["mean_demand_MW"] / 1000

monthly = (m.groupby("month")
             .agg(mean_demand_MW=("demand_MW", "mean"), peak_hourly_MW=("demand_MW", "max"),
                  peak_15min_MW=("demand_peak_MW", "max"), cf_s1=("cf_s1", "mean"),
                  cf_s2=("cf_s2", "mean"), cf_s3=("cf_s3", "mean")).reset_index())

stats = pd.DataFrame([
    ("Annual demand (GWh)", m["demand_MW"].sum() / 1000),
    ("Mean demand (MW)", m["demand_MW"].mean()),
    ("Peak hourly demand (MW)", m["demand_MW"].max()),
    ("Peak 15-min demand (MW)", m["demand_peak_MW"].max()),
    ("Peak apparent power (MVA, 15-min)", m["kva_max"].max() / 1000),
    ("Load factor (mean/peak, hourly)", m["demand_MW"].mean() / m["demand_MW"].max()),
    ("Mean CF scenario 1", m["cf_s1"].mean()), ("Mean CF scenario 2", m["cf_s2"].mean()),
    ("Mean CF scenario 3", m["cf_s3"].mean()), ("Mean CF_t (3-scenario avg)", m["CF_t"].mean()),
],columns=["quantity", "value"])

# Summary of the model parameters that come straight from the datasets (D_t and CF_t).
# Everything else in the parameter table (g, c_b, x0, C_a, battery data, costs, r, ...) is external.
def param_row(symbol, meaning, series, unit, total_label):
    return {"symbol": symbol, "meaning": meaning, "unit": unit, "hours": int(series.size),
            "min": series.min(), "mean": series.mean(), "max": series.max(),
            "annual_total": series.sum(), "annual_total_unit": total_label}

params = pd.DataFrame([
    param_row("D_t", "Hourly campus demand, year 1", m["demand_MW"] * 1000, "kW", "kWh/yr"),
    param_row("CF_t", "PV capacity factor, hourly average of the 3 scenarios", m["CF_t"], "fraction",
              "kWh per kW installed/yr"),
])

with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
    params.to_excel(xw, sheet_name="Parameter_summary", index=False)
    m.rename(columns={"month": "month", "kva_max": "kVA_max_15min"}).to_excel(xw, sheet_name="Hourly", index=False)
    seas.to_excel(xw, sheet_name="Season_summary", index=False)
    monthly.to_excel(xw, sheet_name="Monthly_summary", index=False)
    stats.to_excel(xw, sheet_name="Key_stats", index=False)
    pd.DataFrame(log).to_excel(xw, sheet_name="Cleaning_log", index=False)
    for ws in xw.book.worksheets:              # Arial everywhere, readable widths
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = "0.0000" if ws.title in ("Hourly", "Season_summary", "Monthly_summary") else "0.000"
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = max(12, min(70, max(len(str(c.value or "")) for c in col[:60]) + 2))
        ws.freeze_panes = "A2"

print("\nPARAMETER SUMMARY (from datasets)")
with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 200):
    print(params.to_string(index=False))
print()
print(stats.to_string(index=False))
print(seas.round(3).to_string(index=False))
print("\n".join(f"{r['step']}: {r['detail']}" for r in log))
