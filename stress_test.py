"""
stress_test.py -- resilience analysis: how the designed microgrid copes with random Provincial-grid outages.

Method (two stages):
    1. Design comes from the base model (microgrid_model.py, representative days, min NPV cost). Its PV kW and
       battery modules/energy in each year are read from results/yearly_results.csv and FROZEN here.
    2. For every outage in processed/outage_events.csv a small LP dispatches PV, battery and the reduced grid
       over the outage hours, using the REAL chronological hourly data (model_hourly.csv), not rep days.
       Demand that cannot be met is unserved energy U, charged at VoLL ($/kWh).

Per-outage LP (k = outage hours 0..dur-1, all variables >= 0):
    min   sum_k  VoLL * U_k  +  c_k * G_k  -  c_max * SOC_end   (shed as little as possible, then cheapest grid;
                                                                 energy left in the battery is worth the on-peak price)
    s.t.  G_k + V_k + Pdis_k + U_k = D_k + Pch_k                (energy balance)
          V_k   <= CF_k * PV_y                                  (solar available)
          G_k   <= (1 - f) * G_max     [or (1 - f) * D_k]       (reduced grid during the outage)
          SOC_k  = SOC_{k-1} + mc * Pch_k - Pdis_k / md,  SOC_{-1} = s0 * E_y   (battery bookkeeping)
    s0 = share of the battery that is charged when the outage hits: either fixed (outage_params.json) or, with
         SOC_SOURCE = "base_model", what the base model's battery holds at that hour (see starting_soc below).
          SOC_k <= E_y,  Pch_k, Pdis_k <= pB * N_y              (battery energy / power limits)
          U_k   <= D_k                                          (cannot shed more than the demand)
    D_k = real-hour demand x (1+g)^(y-1); outages are spaced >= 24 h apart, so each one is solved on its own.

Reads:   processed/parameters.json, processed/model_hourly.csv, processed/outage_events.csv,
         processed/outage_params.json, results/yearly_results.csv (only for DESIGN = "base_model"),
         processed/day_map.csv + results/soc_all_years.csv (only for SOC_SOURCE = "base_model")
Writes (results/, tag = design + outage %, e.g. base_model_f50):
    stress_events_<tag>.csv   one row per outage: demand, unserved kWh, cost, battery energy used
    stress_runs_<tag>.csv     one row per run: outages, outage hours, unserved MWh, cost ($ and NPV $)
    stress_summary.csv        one row per tag (re-running a tag replaces its row) -> your comparison table
"""
import json
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

# =============================================================================
# CONFIG
# =============================================================================
ROOT = Path(__file__).resolve().parent
PROC = ROOT / "processed"
OUT = ROOT / "results"

PV_SCENARIO = "cf_y3"     # use the same PV scenario as the base model run ("cf_y1" / "cf_y2" / "cf_y3")
DESIGN = "base_model"     # "base_model": PV + battery per year from results/yearly_results.csv (run microgrid_model.py first)
                          # "grid_only":  existing PV only, no battery (status-quo comparison)
SOC_SOURCE = "base_model" # "base_model": battery charge when an outage hits = base model's SOC at that hour
                          # (real day -> rep day via day_map.csv, year y, end of the previous hour)


# =============================================================================
# DATA
# =============================================================================
def load_inputs():
    val = lambda path: {k: v["value"] for k, v in json.load(open(path)).items()}
    prm, out = val(PROC / "parameters.json"), val(PROC / "outage_params.json")
    hourly = pd.read_csv(PROC / "model_hourly.csv")
    events = pd.read_csv(PROC / "outage_events.csv")
    H = int(prm["horizon_years"])
    years = list(range(1, H + 1))

    if DESIGN == "grid_only":
        design = pd.DataFrame({"year": years, "pv_kW": prm["pv_existing_x0_kW"], "batt_kWh": 0.0, "batt_kW": 0.0})
    else:
        f = OUT / "yearly_results.csv"
        assert f.exists(), f"{f} not found: run microgrid_model.py first (or set DESIGN = 'grid_only')"
        yr = pd.read_csv(f)
        design = pd.DataFrame({"year": yr["year"], "pv_kW": yr["total_PV_kW"], "batt_kWh": yr["batt_energy_kWh"],
                               "batt_kW": yr["batt_in_service"] * prm["battery_power_p_kW"]})
    assert list(design["year"]) == years, "yearly_results.csv must have one row per year 1..H"

    soc = None
    if SOC_SOURCE == "base_model" and DESIGN == "base_model":
        for f in (PROC / "day_map.csv", OUT / "soc_all_years.csv"):
            assert f.exists(), f"{f} not found: re-run prepare_data.py and microgrid_model.py (or set SOC_SOURCE = 'fixed')"
        dm = pd.read_csv(PROC / "day_map.csv")["p"].to_numpy()      # dm[day] = rep day p of real day 0..364
        s = pd.read_csv(OUT / "soc_all_years.csv")
        s["share"] = np.where(s["E_kWh"] > 1e-6, s["SOC_kWh"] / s["E_kWh"].clip(lower=1e-6), 0.0).clip(0, 1)
        soc = dict(day_map=dm, share=s.set_index(["year", "p", "h"])["share"])

    return dict(prm=prm, out=out, events=events, design=design.set_index("year"), soc=soc,
                D=hourly["demand_kw"].to_numpy(), CF=hourly[PV_SCENARIO].to_numpy(),
                c=hourly["price_per_kwh"].to_numpy())


def starting_soc(d, year, start_t):
    """Share of the battery that is charged when an outage starts at real hour start_t of `year`.
    Base model: SOC is stored at the END of each hour, so the charge going into hour h is SOC[p, h-1, y]
    (hour 0 looks at hour 23 of the same rep day, matching the model's cyclic daily SOC)."""
    if d["soc"] is None:
        return d["out"]["battery_soc_at_outage_start"]
    day, h = divmod(int(start_t), 24)
    p = int(d["soc"]["day_map"][day])
    return float(d["soc"]["share"].loc[(year, p, (h - 1) % 24)])


# =============================================================================
# ONE OUTAGE = ONE SMALL LP
# =============================================================================
def solve_event(env, D, S, c, cap, E, Pmax, soc0, mc, md, voll):
    K = range(len(D))
    m = gp.Model(env=env)
    G = m.addVars(K, ub=list(cap), name="G")            # reduced grid connection
    V = m.addVars(K, ub=list(S), name="V")              # solar available this hour
    Pch = m.addVars(K, ub=Pmax, name="Pch")
    Pdis = m.addVars(K, ub=Pmax, name="Pdis")
    SOC = m.addVars(K, ub=E, name="SOC")
    U = m.addVars(K, ub=list(D), name="U")              # unserved demand
    for k in K:
        m.addConstr(G[k] + V[k] + Pdis[k] + U[k] == D[k] + Pch[k])
        m.addConstr(SOC[k] == (soc0 if k == 0 else SOC[k - 1]) + mc * Pch[k] - Pdis[k] / md)
    # leftover battery energy is valued at the highest TOU price, so the battery is not drained for no reason
    # while the reduced grid still has room (does not change unserved energy, only makes the dispatch realistic)
    m.setObjective(gp.quicksum(voll * U[k] + c[k] * G[k] for k in K) - max(c) * SOC[len(D) - 1], GRB.MINIMIZE)
    m.optimize()
    assert m.Status == GRB.OPTIMAL, f"event LP status {m.Status}"

    bal = max(abs(G[k].X + V[k].X + Pdis[k].X + U[k].X - D[k] - Pch[k].X) for k in K)
    assert bal < 1e-4 and all(G[k].X <= cap[k] + 1e-6 for k in K), "event validation failed"
    return {"unserved_kWh": sum(U[k].X for k in K),
            "peak_shortfall_kW": max(U[k].X for k in K),
            "grid_kWh": sum(G[k].X for k in K),
            "solar_kWh": sum(V[k].X for k in K),
            "battery_kWh_discharged": sum(Pdis[k].X for k in K)}


# =============================================================================
# STRESS TEST
# =============================================================================
def run(d):
    prm, out, ev, des = d["prm"], d["out"], d["events"], d["design"]
    f, basis, voll = out["outage_fraction"], out["outage_basis"], out["value_of_lost_load_VoLL"]
    g, r, Gmax = prm["demand_growth_g"], prm["discount_rate_r"], prm["grid_max_G_max_kW"]
    mc, md = prm["battery_charge_eff_mc"], prm["battery_discharge_eff_md"]

    env = gp.Env(params={"OutputFlag": 0})
    rows = []
    for e in ev.itertuples(index=False):
        hrs = np.arange(e.start_t, e.start_t + e.duration_h)
        y, dz = e.year, des.loc[e.year]
        D = d["D"][hrs] * (1 + g) ** (y - 1)                       # real-hour demand, grown to year y
        S = d["CF"][hrs] * dz.pv_kW
        cap = np.full(len(hrs), (1 - f) * Gmax) if basis == "gmax" else (1 - f) * D
        s0 = starting_soc(d, y, e.start_t)
        res = solve_event(env, D, S, d["c"][hrs], cap, dz.batt_kWh, dz.batt_kW, s0 * dz.batt_kWh, mc, md, voll)
        cost = voll * res["unserved_kWh"]
        rows.append({**e._asdict(), "soc_start_share": s0, "demand_kWh": D.sum(), **res,
                     "served_share": 1 - res["unserved_kWh"] / D.sum(),
                     "unserved_cost_$": cost, "unserved_cost_npv_$": cost / (1 + r) ** y})
    env.dispose()
    events = pd.DataFrame(rows)

    runs = events.groupby("run").agg(
        n_outages=("event", "size"), outage_hours=("duration_h", "sum"),
        demand_MWh=("demand_kWh", lambda s: s.sum() / 1e3), unserved_MWh=("unserved_kWh", lambda s: s.sum() / 1e3),
        unserved_cost_=("unserved_cost_$", "sum"), unserved_cost_npv_=("unserved_cost_npv_$", "sum"),
        worst_event_MWh=("unserved_kWh", lambda s: s.max() / 1e3)).reset_index()
    runs = runs.rename(columns={"unserved_cost_": "unserved_cost_$", "unserved_cost_npv_": "unserved_cost_npv_$"})
    runs["served_share"] = 1 - runs["unserved_MWh"] / runs["demand_MWh"]
    return events, runs


def summarize(runs, d, tag):
    out = d["out"]
    n = len(runs)
    stat = lambda col: {f"{col}_mean": runs[col].mean(),
                        f"{col}_ci95": 1.96 * runs[col].std(ddof=1) / np.sqrt(n),   # +/- on the mean
                        f"{col}_p5": runs[col].quantile(0.05), f"{col}_p95": runs[col].quantile(0.95)}
    return {"tag": tag, "design": DESIGN, "outage_fraction": out["outage_fraction"], "basis": out["outage_basis"],
            "VoLL_$/kWh": out["value_of_lost_load_VoLL"],
            "soc_start": "base_model" if d["soc"] is not None else out["battery_soc_at_outage_start"],
            "pv_scenario": PV_SCENARIO, "n_runs": n, "outages_per_run_mean": runs["n_outages"].mean(),
            **stat("unserved_MWh"), **stat("unserved_cost_$"), **stat("unserved_cost_npv_$"),
            "served_share_mean": runs["served_share"].mean(), "worst_event_MWh_max": runs["worst_event_MWh"].max()}


# =============================================================================
# MAIN
# =============================================================================
def main():
    d = load_inputs()
    out, des = d["out"], d["design"]
    tag = f"{DESIGN}_f{round(100 * out['outage_fraction'])}" + ("_socfixed" if SOC_SOURCE == "fixed" and DESIGN == "base_model" else "")
    print(f"Design: {DESIGN}   year 1: {des.loc[1].pv_kW:,.0f} kW PV, {des.loc[1].batt_kWh:,.0f} kWh battery"
          f"   year {des.index[-1]}: {des.iloc[-1].pv_kW:,.0f} kW PV, {des.iloc[-1].batt_kWh:,.0f} kWh battery")
    print(f"Outages: {len(d['events'])} events in {d['events'].run.nunique()} runs, "
          f"{out['outage_fraction']:.0%} of capacity lost ({out['outage_basis']} basis), VoLL ${out['value_of_lost_load_VoLL']}/kWh")

    events, runs = run(d)
    s = summarize(runs, d, tag)

    OUT.mkdir(exist_ok=True)
    events.to_csv(OUT / f"stress_events_{tag}.csv", index=False)
    runs.to_csv(OUT / f"stress_runs_{tag}.csv", index=False)
    summ_path = OUT / "stress_summary.csv"
    old = pd.read_csv(summ_path) if summ_path.exists() else pd.DataFrame(columns=["tag"])
    summ = pd.concat([old[old["tag"] != tag], pd.DataFrame([s])], ignore_index=True)   # replace this tag's row
    summ.to_csv(summ_path, index=False)

    print(f"\nPer 25-year run ({s['n_runs']} runs):          mean     +/-95%CI        P5       P95")
    for col, unit in [("unserved_MWh", "MWh"), ("unserved_cost_$", "$"), ("unserved_cost_npv_$", "$")]:
        print(f"  {col + ' (' + unit + ')':28s} {s[col + '_mean']:>12,.0f} {s[col + '_ci95']:>10,.0f}"
              f" {s[col + '_p5']:>12,.0f} {s[col + '_p95']:>12,.0f}")
    print(f"  share of outage-hour demand served   {s['served_share_mean']:.1%}")
    print(f"  battery charge when outages hit      mean {events['soc_start_share'].mean():.0%}"
          f"  ({'base model SOC' if d['soc'] is not None else 'fixed'};"
          f" {(events['soc_start_share'] < 0.1).mean():.0%} of outages find it under 10%)")
    print(f"  worst single outage                  {s['worst_event_MWh_max']:,.1f} MWh unserved")
    by_year = events.groupby("year")["unserved_kWh"].sum() / 1e3 / s["n_runs"]
    print("\nMean unserved MWh per run, by year:", by_year.reindex(des.index, fill_value=0).round(1).to_dict())
    print(f"\nWrote stress_events_{tag}.csv, stress_runs_{tag}.csv and updated stress_summary.csv in {OUT}")


if __name__ == "__main__":
    main()
