"""
prepare_data.py -- load, align and reduce the UW demand + PV data, and write the model parameters.

Reads (Data/):
    DemandProfile_UW.csv   15-min campus demand (columns DATE, TIME, kW, ..)
    hourly_data.csv        hourly PV capacity factors (columns Month, Day, Hour, Year 1/2/3 capacity factor %)

Writes (processed/):
    model_hourly.csv   8760 rows, Jan 1 -> Dec 31: demand_kw, cf_y1, cf_y2, cf_y3, tou_bucket, price_per_kwh
    rep_periods.csv    K representative days x 24 h: same columns + p, h, weight (days represented; sum = 365)
    parameters.json    every model parameter as {value, unit, source} for the Gurobi script
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# =============================================================================
# CONFIG
# =============================================================================
ROOT = Path(__file__).resolve().parent
DEMAND_CSV = ROOT / "Data" / "DemandProfile_UW.csv"
PV_CSV = ROOT / "Data" / "SolarPVData.csv"
OUT_DIR = ROOT / "processed"

K_REP_DAYS = 12        # representative days for the k-means version
K_SEQ_DAYS = 12        # periods for the sequential version (12 = one per calendar month)
SEQ_ADD_PEAK_DAY = False   # True -> extra 1-day period for the peak-demand day (K_SEQ_DAYS + 1 periods)

# OEB TOU prices ($/kWh) Nov 1 2026 - Oct 31 2027: off-peak, mid-peak, on-peak.
TOU_PRICE = [0.091, 0.146, 0.189]

SCEN = ["cf_y1", "cf_y2", "cf_y3"] # the three PV scenarios
SITES = ["F", "C", "H", "L", "O", "X", "V", "S", "K", "M", "W"] # ground field + 9 parking lots
HORIZON = 25  # planning years


# =============================================================================
# PARAMETERS
# =============================================================================
def P(value, unit, source="PLACEHOLDER"):
    return {"value": value, "unit": unit, "source": source}


PARAMS = {
    # general
    "horizon_years":              P(HORIZON, "years", "model decision"), #25 years
    "discount_rate_r":            P(0.04, "1/yr", "common estimate"), #0.04
    "demand_growth_g":            P(0.02, "1/yr", "campus plan"), #0.02
    "grid_max_G_max_kW":          P(22000.0, "kW", 'assumed'), #22000

    # PV
    "pv_existing_x0_kW":          P(60, "kW", 'doc'), #60
    "pv_site_capacity_C_a_kW":    P({"F": 10308,
                                      "C": 2485,
                                      "H": 736, 
                                      "L": 635, 
                                      "O": 660, 
                                      "X": 2052, 
                                      "V": 227, 
                                      "S": 228, 
                                      "K": 273, 
                                      "M": 615,
                                      "W": 522}, "kW", "Calculations"),
    "pv_site_fixed_cost_F_a":     P({"F": 4276258,
                                      "C":797827,
                                      "H":327005, 
                                      "L":189957, 
                                      "O":377685, 
                                      "X":715685, 
                                      "V":161302, 
                                      "S":126957, 
                                      "K":170422, 
                                      "M":203678,
                                      "W":212828}, "$", "Calculations"), 
    "pv_cost_year1_per_kW":       P({"F":1250, #1250
                                      "C":2904, #2904
                                      "H":2904, #2904
                                      "L":2904, #2904
                                      "O":2904, #2904
                                      "X":2904, #2904
                                      "V":2904, #2904
                                      "S":2904, #2904
                                      "K":2904, #2904
                                      "M": 2904,#2904
                                      "W": 2904}, "$/kW", "Calculations"), # #2904
    "pv_cost_annual_change":      P(0, "1/yr", "ignored" ), # 0
    "pv_om_per_kW_yr":            P(21, "$/kW/yr", "website"), #21
    "pv_life_years":              P(25, "years", 'researched'), # 25

    # battery module
    "battery_energy_e_kWh":       P(514, "kWh/module", 'researched'), #514
    "battery_power_p_kW":         P(125, "kW/module", 'researched'), #125
    "battery_max_modules_N_max":  P(78, "modules", "based on UCSD"), #78
    "battery_charge_eff_mc":      P(0.95, "-", 'researched'), #0.95
    "battery_discharge_eff_md":   P(0.95, "-", 'researched'), #0.95
    "battery_life_L_years":       P(15, "years", 'researched'),#15
    "battery_fade_per_year":      P(0, "1/yr", 'researched'), #0
    "battery_cost_year1":         P(212000, "$/module", 'tesla battery'), # 212000
    "battery_cost_annual_change": P(0, "1/yr", 'ignored'), #0
    "battery_om_per_module_yr":   P(5300, "$/module/yr", 'researched'), #5300
}


def derive_params(params, df):
    """Year-by-year arrays the Gurobi model needs, built from the entries above plus the data.
    Year-indexed lists: element 0 = year 1. q_s list: element s = share of capacity left at age s."""
    v = lambda k: params[k]["value"]
    years = range(1, HORIZON + 1)
    src = "DERIVED from the parameters"
    d = {}

    d["pv_cost_K_PV_a_y"] = P({a: [v("pv_cost_year1_per_kW")[a] * (1 + v("pv_cost_annual_change")) ** (y - 1)
                                   for y in years] for a in SITES}, "$/kW", src)
    d["battery_cost_K_B_y"] = P([v("battery_cost_year1") * (1 + v("battery_cost_annual_change")) ** (y - 1)
                                 for y in years], "$/module", src)
    d["battery_capacity_share_q_s"] = P([max(0.0, 1 - v("battery_fade_per_year") * s)
                                         for s in range(v("battery_life_L_years"))], "-", src)
    # salvage: straight-line remaining-life fraction at the end of the horizon (age = HORIZON - y + 1).
    d["pv_salvage_sigma_PV_y"] = P([max(0.0, 1 - (HORIZON - y + 1) / v("pv_life_years")) for y in years], "-", src)
    d["battery_salvage_sigma_B_y"] = P([max(0.0, 1 - (HORIZON - y + 1) / v("battery_life_L_years"))
                                        for y in years], "-", src)

    # from the data / OEB
    d["tou_price_c_b"] = P(TOU_PRICE * 2, "$/kWh", "OEB RPP")
    d["data_summary"] = P({
        "n_hours": int(len(df)),
        "demand_annual_kWh": float(df["demand_kw"].sum()),
        "demand_peak_hourly_kW": float(df["demand_kw"].max()),
        "demand_mean_kW": float(df["demand_kw"].mean()),
        "pv_annual_kWh_per_kW": {s: float(df[s].sum()) for s in SCEN},
    }, "-", "computed from the data files")
    return d


# =============================================================================
# LOAD + ALIGN
# =============================================================================
def load_demand(path):
    """15-min interval-ENDING demand -> hourly mean kW indexed by hour start."""
    # read as text; title row above the header and ragged rows
    raw = pd.read_csv(path, header=None, names=range(12), dtype=str, skip_blank_lines=True, encoding="utf-8-sig")
    raw = raw.dropna(axis=1, how="all")
    first = raw.iloc[:, 0].fillna("").str.strip().str.upper()
    hdr = first[first == "DATE"].index[0]
    body = raw.iloc[hdr + 1:].copy()
    body.columns = [str(c).strip().lower() for c in raw.iloc[hdr]]
    body = body.dropna(subset=["date"])

    def to_td(s):
        hh, mm = str(s).strip().split(":")[:2]
        td = pd.Timedelta(hours=int(hh), minutes=int(mm))
        return td if td > pd.Timedelta(0) else pd.Timedelta(hours=24)   # a 00:00 stamp is end-of-day (24:00)

    end = pd.to_datetime(body["date"], format="mixed").dt.normalize() + body["time"].apply(to_td)
    q = pd.Series(pd.to_numeric(body["kw"]).values, index=pd.DatetimeIndex(end))
    assert len(q) == 35040 and q.notna().all(), f"expected 35,040 quarter-hour rows, got {len(q)}"
    assert (q.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all(), \
        "demand timestamps are not a clean 15-minute sequence (check the date format / midnight rows)"

    hourly = q.set_axis(q.index - pd.Timedelta(minutes=15)).resample("h").mean()   # stamps -> interval start
    assert len(hourly) == 8760
    return hourly.rename("demand_kw").rename_axis("timestamp")


def load_pv(path):
    """Hourly capacity factors for the three PV years (fractions 0-1). Blank hours are filled with the
    mean of the other years' value for that hour."""
    # read as text; tolerate a title row, a BOM, and blank filler rows/columns (",,,,," lines from Excel)
    raw = pd.read_csv(path, header=None, names=range(12), dtype=str, encoding="utf-8-sig")
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all").reset_index(drop=True)
    first = raw.iloc[:, 0].fillna("").str.strip().str.lower()
    if not (first == "month").any():
        raise ValueError(f"{path.name}: no header row starting with 'Month' found. First rows are:\n{raw.head()}")
    hdr = first[first == "month"].index[0]

    pv = raw.iloc[hdr + 1:, :6].copy()
    pv.columns = ["month", "day", "hour_label"] + SCEN
    pv = pv.apply(lambda c: pd.to_numeric(c.astype(str).str.replace("%", ""), errors="coerce"))
    pv = pv.dropna(subset=["month", "day", "hour_label"]).reset_index(drop=True)
    if len(pv) != 8760 or not pv["hour_label"].between(1, 24).all():
        raise ValueError(
            f"{path.name} should be the PV file: Month, Day, Hour, then three capacity-factor (%) columns, "
            f"8760 rows.\nFound {len(pv)} data rows; header = {list(raw.iloc[hdr])}\n{pv.head()}")

    cf = (pv[SCEN] / 100).clip(upper=1.0)           # % -> fraction, cap at 100%
    for c in SCEN:
        pv[c] = cf[c].fillna(cf.drop(columns=c).mean(axis=1))
    pv["hour"] = pv["hour_label"] - 1               # labels 1-24 (hour-ending) -> hour start 0-23
    return pv[["month", "day", "hour"] + SCEN]


def add_tou(df):
    """TOU bucket 0-5 and price per hour (OEB schedule, local clock time). Weekends treated as weekdays."""
    local = pd.DatetimeIndex(df["timestamp"])
    summer = np.asarray((local.month >= 5) & (local.month <= 10))
    h = np.asarray(local.hour)
    morning, midday, evening = (h >= 7) & (h < 11), (h >= 11) & (h < 17), (h >= 17) & (h < 19)
    on = np.where(summer, midday, morning | evening)        # winter on: 7-11, 17-19 | summer on: 11-17
    mid = np.where(summer, morning | evening, midday)       # winter mid: 11-17      | summer mid: 7-11, 17-19
    period = np.where(on, 2, np.where(mid, 1, 0))           # 0 off, 1 mid, 2 on
    df["tou_bucket"] = period + 3 * summer
    df["price_per_kwh"] = np.array(TOU_PRICE)[period]
    return df


def build_hourly():
    demand = load_demand(DEMAND_CSV)
    pv = load_pv(PV_CSV)

    # demand is Oct 2025 - Sep 2026; PV is a Jan-Dec typical year -> match on (month, day, hour)
    df = demand.reset_index()
    df["month"], df["day"], df["hour"] = df.timestamp.dt.month, df.timestamp.dt.day, df.timestamp.dt.hour
    df = df.merge(pv, on=["month", "day", "hour"], how="left", validate="one_to_one")
    assert df[SCEN].notna().all().all(), "PV/demand alignment left gaps"
    df = df.sort_values(["month", "day", "hour"]).reset_index(drop=True)    # Jan 1 -> Dec 31
    df.index.name = "t"
    df = add_tou(df)
    return df[["timestamp", "demand_kw"] + SCEN + ["tou_bucket", "price_per_kwh"]]


# =============================================================================
# REPRESENTATIVE DAYS
# =============================================================================
def representative_days(df, k):
    """k representative days. Demand and PV = cluster average (annual totals preserved exactly);
    the peak-demand day is forced in as its own real day; price/bucket come from the cluster's medoid."""
    cols = ["demand_kw", "price_per_kwh", "tou_bucket"] + SCEN
    M = {c: df[c].values.reshape(365, 24) for c in cols}
    z = lambda a: (a - a.mean()) / a.std()
    X = np.hstack([z(M["demand_kw"]), z(M["price_per_kwh"])] + [z(M[c]) / np.sqrt(3) for c in SCEN])

    peak = int(M["demand_kw"].max(axis=1).argmax())
    rest = np.array([i for i in range(365) if i != peak])
    label = np.full(365, k - 1) # cluster k-1 = the peak day alone
    label[rest] = KMeans(k - 1, n_init=50, random_state=0).fit(X[rest]).labels_

    parts = []
    for c in range(k):
        m = np.where(label == c)[0]
        med = m[np.argmin(np.linalg.norm(X[m] - X[m].mean(axis=0), axis=1))]
        parts.append((med, pd.DataFrame({
            "h": range(24), "weight": len(m),
            "demand_kw": M["demand_kw"][m].mean(axis=0),
            **{s: M[s][m].mean(axis=0) for s in SCEN},
            "tou_bucket": M["tou_bucket"][med].astype(int),
            "price_per_kwh": M["price_per_kwh"][med]})))
    parts.sort(key=lambda x: x[0])
    rep = pd.concat([d.assign(p=i) for i, (_, d) in enumerate(parts)], ignore_index=True)
    return rep[["p", "h", "weight", "demand_kw"] + SCEN + ["tou_bucket", "price_per_kwh"]]

def representative_days_sequential(df, k=12, add_peak_day=False):
    """k representative days built from consecutive blocks of the year (no clustering).
    k == 12 -> one rep day per calendar month (weights = 31, 28, 31, ...).
    other k -> k contiguous, near-equal blocks of the 365 days (np.array_split).
    Demand and PV = block average (annual totals preserved exactly); price/bucket come from the
    block's medoid day (identical for every day in a month, since the TOU schedule is seasonal).
    add_peak_day=True removes the peak-demand day from its block and appends it as an extra
    period (k + 1 periods total, weight 1), so the peak isn't averaged away."""
    cols = ["demand_kw", "price_per_kwh", "tou_bucket"] + SCEN
    M = {c: df[c].values.reshape(365, 24) for c in cols}

    if k == 12:
        month = df["timestamp"].dt.month.values.reshape(365, 24)[:, 0]   # month of each day (rows are Jan 1 -> Dec 31)
        groups = [np.where(month == m)[0] for m in range(1, 13)]
    else:
        groups = np.array_split(np.arange(365), k)

    if add_peak_day:
        peak = int(M["demand_kw"].max(axis=1).argmax())
        groups = [g[g != peak] for g in groups] + [np.array([peak])]

    parts = []
    for p, m in enumerate(groups):
        price = M["price_per_kwh"][m]
        med = m[np.argmin(np.linalg.norm(price - price.mean(axis=0), axis=1))]   # most representative price pattern
        parts.append(pd.DataFrame({
            "p": p, "h": range(24), "weight": len(m),
            "demand_kw": M["demand_kw"][m].mean(axis=0),
            **{s: M[s][m].mean(axis=0) for s in SCEN},
            "tou_bucket": M["tou_bucket"][med].astype(int),
            "price_per_kwh": M["price_per_kwh"][med]}))
    rep = pd.concat(parts, ignore_index=True)
    return rep[["p", "h", "weight", "demand_kw"] + SCEN + ["tou_bucket", "price_per_kwh"]]


def check(df, rep):
    w = rep["weight"]
    rows = {"annual demand (GWh)": (df.demand_kw.sum() / 1e6, (w * rep.demand_kw).sum() / 1e6),
            "grid-only cost ($M)": ((df.demand_kw * df.price_per_kwh).sum() / 1e6,
                                    (w * rep.demand_kw * rep.price_per_kwh).sum() / 1e6),
            "peak demand (MW)": (df.demand_kw.max() / 1e3, rep.demand_kw.max() / 1e3)}
    rows.update({f"{s} yield (kWh/kW)": (df[s].sum(), (w * rep[s]).sum()) for s in SCEN})
    out = pd.DataFrame(rows, index=["full year", "rep. days"]).T
    out["error %"] = 100 * (out["rep. days"] / out["full year"] - 1)
    print(out.round(3).to_string())


# =============================================================================
# MAIN
# =============================================================================
def main():
    OUT_DIR.mkdir(exist_ok=True)

    df = build_hourly()
    rep = representative_days(df, K_REP_DAYS)
    rep_seq = representative_days_sequential(df, K_SEQ_DAYS, add_peak_day=SEQ_ADD_PEAK_DAY)

    df.to_csv(OUT_DIR / "model_hourly.csv")
    rep.to_csv(OUT_DIR / "rep_periods.csv", index=False)
    rep_seq.to_csv(OUT_DIR / "rep_periods_sequential.csv", index=False)

    params = {**PARAMS, **derive_params(PARAMS, df)}
    with open(OUT_DIR / "parameters.json", "w") as f:
        json.dump(params, f, indent=2)

    print(f"Saved model_hourly.csv ({len(df)} rows), rep_periods.csv ({K_REP_DAYS} days), "
          f"rep_periods_sequential.csv ({rep_seq['p'].nunique()} periods) "
          f"and parameters.json to {OUT_DIR}\n")
    print("K-means representative days:")
    check(df, rep)
    print("\nSequential representative days:")
    check(df, rep_seq)


if __name__ == "__main__":
    main()