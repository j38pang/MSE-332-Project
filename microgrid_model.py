"""
microgrid_model.py -- MSE 332 base model (UW microgrid: PV sites + Megapack BESS + Ontario grid), Gurobi MILP.

Follows the formulation in the progress report:
  sets      t (hours), y (years 1..25), a (PV sites), b (TOU buckets)
  design    x[a,y] >= 0 kW PV bought, z[a,y] in {0,1} site opened, n[y] in Z+ battery modules bought
  operation G grid import, V solar used, Pch/Pdis battery power, SOC stored energy
  objective min NPV of grid energy + PV/BESS capex + fixed site costs + O&M - salvage

Time resolution: the 12 representative days in processed/rep_periods.csv (24 h each, `weight` = number of
real days each one stands for, weights sum to 365). Every hourly sum over the year becomes
    sum_t f_t  ->  sum_p weight_p * sum_h f_{p,h}
and the battery state of charge is cyclic within each representative day (SOC at the end of hour 23 = SOC
before hour 0), which replaces the report's yearly cycle SOC_0 = SOC_|T|.
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

PV_SCENARIO = "cf_y3"          # "avg" = mean of the 3 EV3 years (report assumption), or "cf_y1" / "cf_y2" / "cf_y3"
OBJECTIVE = "cost"           # "cost" = min total NPV cost (Z);  "grid" = min total grid energy (w)
STRICT_SITE_OPENING = True   # True: report constraint (b) x[a,y] <= C_a z[a,y] (a site is built out in the year it opens)
                             # False: x[a,y] <= C_a * (sum_{j<=y} z[a,j]) (PV can be added to an opened site later)
TIME_LIMIT_S = 600
MIP_GAP = 0.005



# =============================================================================
# DATA
# =============================================================================
def load_inputs():
    with open(PROC / "parameters.json") as f:
        raw = json.load(f)
    prm = {k: v["value"] for k, v in raw.items()}
    rep = pd.read_csv(PROC / "rep_periods.csv")

    H = int(prm["horizon_years"])
    years = list(range(1, H + 1))
    sites = list(prm["pv_site_capacity_C_a_kW"].keys())
    periods = sorted(int(p) for p in rep["p"].unique())
    hours = list(range(24))

    rep = rep.set_index(["p", "h"]).sort_index()
    cf = rep[["cf_y1", "cf_y2", "cf_y3"]].mean(axis=1) if PV_SCENARIO == "avg" else rep[PV_SCENARIO]

    KB = prm["battery_cost_K_B_y"]
    d = dict(
        years=years, sites=sites, periods=periods, hours=hours, H_full=int(prm["horizon_years"]),
        w={p: float(rep.loc[(p, 0), "weight"]) for p in periods},
        D={(p, h): float(rep.loc[(p, h), "demand_kw"]) for p in periods for h in hours},
        CF={(p, h): float(cf.loc[(p, h)]) for p in periods for h in hours},
        c={(p, h): float(rep.loc[(p, h), "price_per_kwh"]) for p in periods for h in hours},
        g=prm["demand_growth_g"], r=prm["discount_rate_r"], Gmax=prm["grid_max_G_max_kW"], x0=prm["pv_existing_x0_kW"],
        C=prm["pv_site_capacity_C_a_kW"], F=prm["pv_site_fixed_cost_F_a"],
        KPV={a: prm["pv_cost_K_PV_a_y"][a] for a in sites},
        oPV=prm["pv_om_per_kW_yr"],
        e=prm["battery_energy_e_kWh"], pB=prm["battery_power_p_kW"], Nmax=prm["battery_max_modules_N_max"],
        mc=prm["battery_charge_eff_mc"], md=prm["battery_discharge_eff_md"],
        L=int(prm["battery_life_L_years"]), q=prm["battery_capacity_share_q_s"],
        KB=KB, oB=prm["battery_om_per_module_yr"],
        sPV=prm["pv_salvage_sigma_PV_y"], sB=prm["battery_salvage_sigma_B_y"],
    )
    assert abs(sum(d["w"].values()) - 365) < 1e-6, "representative-day weights should sum to 365"
    return d


# =============================================================================
# MODEL
# =============================================================================
def build(d):
    Y, A, Pp, Hh = d["years"], d["sites"], d["periods"], d["hours"]
    T = [(p, h) for p in Pp for h in Hh]
    L, H = d["L"], d["H_full"]
    m = gp.Model("UW_microgrid")

    # design variables
    x = m.addVars(A, Y, lb=0, name="x")                       # PV kW bought at site a in year y
    z = m.addVars(A, Y, vtype=GRB.BINARY, name="z")           # site a opened in year y
    n = m.addVars(Y, lb=0, ub=d["Nmax"], vtype=GRB.INTEGER, name="n")  # battery modules bought in year y
    # operating variables (every rep hour, every year)
    G = m.addVars(T, Y, lb=0, ub=d["Gmax"], name="G")         # grid import kW (ub = grid connection limit)
    V = m.addVars(T, Y, lb=0, name="V")                       # solar used kW
    Pch = m.addVars(T, Y, lb=0, name="Pch")
    Pdis = m.addVars(T, Y, lb=0, name="Pdis")
    SOC = m.addVars(T, Y, lb=0, name="SOC")                   # kWh at the end of hour h
    # capacity bookkeeping (defined by equality constraints)
    X = m.addVars(Y, lb=0, name="X")                          # cumulative new PV kW
    N = m.addVars(Y, lb=0, name="N")                          # battery modules in service
    E = m.addVars(Y, lb=0, name="E")                          # faded battery energy capacity kWh

    alive = {y: [j for j in Y if max(1, y - L + 1) <= j <= y] for y in Y}
    for y in Y:
        m.addConstr(X[y] == gp.quicksum(x[a, j] for a in A for j in Y if j <= y), name=f"Xdef[{y}]")
        m.addConstr(N[y] == gp.quicksum(n[j] for j in alive[y]), name=f"Ndef[{y}]")
        m.addConstr(E[y] == d["e"] * gp.quicksum(d["q"][y - j] * n[j] for j in alive[y]), name=f"Edef[{y}]")
        m.addConstr(N[y] <= d["Nmax"], name=f"Nmax[{y}]")

    # PV site opening and area limits, report
    for a in A:
        Ca = d["C"][a]
        m.addConstr(x.sum(a, "*") <= Ca * z.sum(a, "*"), name=f"siteArea[{a}]")          # (a)
        m.addConstr(z.sum(a, "*") <= 1, name=f"openOnce[{a}]")                           # (c)
        for y in Y:
            rhs = Ca * z[a, y] if STRICT_SITE_OPENING else Ca * gp.quicksum(z[a, j] for j in Y if j <= y)
            m.addConstr(x[a, y] <= rhs, name=f"siteOpen[{a},{y}]")                       # (b)

    # hourly operation
    for y in Y:
        grow = (1 + d["g"]) ** (y - 1)
        for p in Pp:
            for h in Hh:
                t = (p, h)
                m.addConstr(G[(*t, y)] + V[(*t, y)] + Pdis[(*t, y)] == d["D"][t] * grow + Pch[(*t, y)], name=f"bal[{p},{h},{y}]")
                m.addConstr(V[(*t, y)] <= d["CF"][t] * (d["x0"] + X[y]), name=f"pvAvail[{p},{h},{y}]")
                prev = (p, 23) if h == 0 else (p, h - 1)                    # cyclic within the rep day
                m.addConstr(SOC[(*t, y)] == SOC[(*prev, y)] + d["mc"] * Pch[(*t, y)] - Pdis[(*t, y)] / d["md"],
                            name=f"soc[{p},{h},{y}]")
                m.addConstr(SOC[(*t, y)] <= E[y], name=f"socMax[{p},{h},{y}]")
                m.addConstr(Pch[(*t, y)] <= d["pB"] * N[y], name=f"chMax[{p},{h},{y}]")
                m.addConstr(Pdis[(*t, y)] <= d["pB"] * N[y], name=f"disMax[{p},{h},{y}]")

    # objectives
    disc = {y: 1 / (1 + d["r"]) ** y for y in Y}
    grid_cost = {y: gp.quicksum(d["w"][p] * d["c"][(p, h)] * G[p, h, y] for p in Pp for h in Hh) for y in Y}
    yearly = {y: grid_cost[y]
                 + gp.quicksum(d["KPV"][a][y - 1] * x[a, y] + d["F"][a] * z[a, y] for a in A)
                 + d["KB"][y - 1] * n[y]
                 + d["oPV"] * X[y] + d["oB"] * N[y] for y in Y}
    # salvage: straight-line remaining life at the end of the horizon (per purchase year, from parameters.json)
    salvage = (1 / (1 + d["r"]) ** H) * gp.quicksum(
        d["sPV"][y - 1] * d["KPV"][a][y - 1] * x[a, y] for a in A for y in Y) + \
        (1 / (1 + d["r"]) ** H) * gp.quicksum(d["sB"][y - 1] * d["KB"][y - 1] * n[y] for y in Y)
    Z = gp.quicksum(disc[y] * yearly[y] for y in Y) - salvage
    w_grid = gp.quicksum(d["w"][p] * G[p, h, y] for p in Pp for h in Hh for y in Y)

    m.setObjective(Z if OBJECTIVE == "cost" else w_grid, GRB.MINIMIZE)
    m.Params.TimeLimit, m.Params.MIPGap = TIME_LIMIT_S, MIP_GAP

    v = dict(x=x, z=z, n=n, G=G, V=V, Pch=Pch, Pdis=Pdis, SOC=SOC, X=X, N=N, E=E)
    exprs = dict(Z=Z, w_grid=w_grid, grid_cost=grid_cost, salvage=salvage)
    return m, v, exprs


# =============================================================================
# RESULTS + VALIDATION
# =============================================================================
def report(m, v, ex, d):
    if m.SolCount == 0:
        print(f"No solution (status {m.Status}).")
        if m.Status == GRB.INFEASIBLE:
            m.computeIIS(); m.write(str(OUT / "infeasible.ilp"))
            print("IIS written to results/infeasible.ilp")
        return
    Y, A, Pp, Hh = d["years"], d["sites"], d["periods"], d["hours"]
    T = [(p, h) for p in Pp for h in Hh]
    W = lambda f: sum(d["w"][p] * f(p, h) for p, h in T)        # annual total from rep days

    rows = []
    for y in Y:
        grow = (1 + d["g"]) ** (y - 1)
        rows.append({
            "year": y,
            "new_PV_kW": sum(v["x"][a, y].X for a in A),
            "total_PV_kW": d["x0"] + v["X"][y].X,
            "sites_opened": " ".join(a for a in A if v["z"][a, y].X > 0.5),
            "batt_bought": round(v["n"][y].X),
            "batt_in_service": round(v["N"][y].X),
            "batt_energy_kWh": v["E"][y].X,
            "demand_MWh": W(lambda p, h: d["D"][(p, h)] * grow) / 1e3,
            "grid_MWh": W(lambda p, h: v["G"][p, h, y].X) / 1e3,
            "solar_used_MWh": W(lambda p, h: v["V"][p, h, y].X) / 1e3,
            "batt_discharge_MWh": W(lambda p, h: v["Pdis"][p, h, y].X) / 1e3,
            "grid_peak_kW": max(v["G"][(*t, y)].X for t in T),
            "grid_cost_$": ex["grid_cost"][y].getValue(),
        })
    res = pd.DataFrame(rows)

    # validation: energy balance, SOC limits, power limits, grid limit
    tol = 1e-4
    bal = max(abs(v["G"][(*t, y)].X + v["V"][(*t, y)].X + v["Pdis"][(*t, y)].X
                  - d["D"][t] * (1 + d["g"]) ** (y - 1) - v["Pch"][(*t, y)].X) for t in T for y in Y)
    socx = max(v["SOC"][(*t, y)].X - v["E"][y].X for t in T for y in Y)
    pwr = max(max(v["Pch"][(*t, y)].X, v["Pdis"][(*t, y)].X) - d["pB"] * v["N"][y].X for t in T for y in Y)
    grid = max(v["G"][(*t, y)].X for t in T for y in Y) - d["Gmax"]
    simul = sum(1 for t in T for y in Y if v["Pch"][(*t, y)].X > 1 and v["Pdis"][(*t, y)].X > 1)

    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "yearly_results.csv", index=False)
    pd.DataFrame([{"site": a, "year_opened": next((y for y in Y if v["z"][a, y].X > 0.5), None),
                   "PV_kW": sum(v["x"][a, y].X for y in Y), "capacity_kW": d["C"][a]} for a in A]
                 ).to_csv(OUT / "pv_sites.csv", index=False)
    # one year of rep-day dispatch (year 1 and final year) for storage-trajectory plots
    disp = [{"year": y, "p": p, "h": h, "weight": d["w"][p],
             "demand_kW": d["D"][(p, h)] * (1 + d["g"]) ** (y - 1), "grid_kW": v["G"][p, h, y].X,
             "solar_kW": v["V"][p, h, y].X, "charge_kW": v["Pch"][p, h, y].X,
             "discharge_kW": v["Pdis"][p, h, y].X, "SOC_kWh": v["SOC"][p, h, y].X}
            for y in (Y[0], Y[-1]) for p, h in T]
    pd.DataFrame(disp).to_csv(OUT / "dispatch_first_last_year.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\nObjective ({OBJECTIVE}): {m.ObjVal:,.0f}   gap {m.MIPGap:.3%}   runtime {m.Runtime:.1f}s")
    print(f"Total NPV cost Z = ${ex['Z'].getValue():,.0f}  (salvage credit ${ex['salvage'].getValue():,.0f})")
    print(f"Total grid energy over horizon = {ex['w_grid'].getValue() / 1e6:,.1f} GWh\n")
    print(res.round(1).to_string(index=False))
    print("\nValidation (should all be ~0 or negative):")
    print(f"  max energy-balance residual   {bal:.2e} kW")
    print(f"  max SOC above capacity        {socx:.2e} kWh")
    print(f"  max power above battery limit {pwr:.2e} kW")
    print(f"  max grid above connection     {grid:.2e} kW")
    print(f"  hours charging & discharging at once: {simul}")
    print(f"\nWrote yearly_results.csv, pv_sites.csv, dispatch_first_last_year.csv to {OUT}")
    assert bal < tol and socx < tol and pwr < tol and grid < tol, "validation failed"


def main():
    d = load_inputs()
    peak_last = max(d["D"].values()) * (1 + d["g"]) ** (d["H_full"] - 1)
    print(f"Growth {d['g']:.1%}/yr -> final-year peak {peak_last:,.0f} kW vs grid limit {d['Gmax']:,.0f} kW")
    m, v, ex = build(d)
    m.optimize()
    OUT.mkdir(exist_ok=True)
    report(m, v, ex, d)


if __name__ == "__main__":
    main()
