#!/usr/bin/env python3
"""Fit the motor and propeller constants from the sweep summaries and the coast data.
掃引のまとめと惰性のデータから、モータとプロペラの定数を当てはめる。

  K_e, R (, V_b) : <v_m> = V_b + R <i> + K_e w      (with- and no-prop steps together / プロペラ有無の全段)
  I_0(w)         : no-prop current = a + b w^2      (no-prop steps / プロペラなしの段)
  C_T            : F = C_T w^2                      (least squares through the origin / 原点を通る最小二乗)
  C_Q            : K_e (I - I_0(w)) = C_Q w^2
  J              : J dw/dt = -(C_Q w^2 + K_e I_0(w)), integrated and fitted to w(t) after motor stop
                   モータ停止後の w(t) に、運動方程式の数値解を当てはめる
Inputs (default ./data/2026-10-15): summary.json, summary_noprop.json, coast_omega.csv
Usage: python3 tools/fit_constants.py [--data DIR]
"""
import argparse, csv, json, pathlib
import numpy as np
from scipy.optimize import minimize_scalar

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='data/2026-10-15')
D = pathlib.Path(ap.parse_args().data)
P = json.loads((D / 'summary.json').read_text())
NP = json.loads((D / 'summary_noprop.json').read_text())

# K_e, R from all steps / 全段から K_e と R
V = np.r_[[x['Vm'] for x in NP], [p['Vm_avg'] for p in P]]
I = np.r_[[x['I'] for x in NP], [p['Ibat'] for p in P]]
w = np.r_[[x['omega'] for x in NP], [p['omega'] for p in P]]
R, Ke = np.linalg.lstsq(np.c_[I, w], V, rcond=None)[0]
Vb, R3, Ke3 = np.linalg.lstsq(np.c_[np.ones_like(I), I, w], V, rcond=None)[0]
print(f'K_e = {Ke:.4e} V s/rad, R = {R:.3f} ohm   (with brush drop: V_b = {Vb:.3f} V, R = {R3:.3f}, K_e = {Ke3:.4e})')

# I_0(w) from no-prop steps / プロペラなしから無負荷電流
wn = np.array([x['omega'] for x in NP]); In = np.array([x['I'] for x in NP])
a0, b0 = np.linalg.lstsq(np.c_[np.ones_like(wn), wn ** 2], In, rcond=None)[0]
print(f'I_0(w) = {a0:.4f} + {b0:.3e} w^2  A')

# C_T, C_Q from with-prop steps / プロペラ付きから C_T と C_Q
wp = np.array([p['omega'] for p in P]); F = np.array([p['thrust_g'] for p in P]) * 9.80665e-3
Ip = np.array([p['Ibat'] for p in P]); tq = Ke * (Ip - (a0 + b0 * wp ** 2))
CT = np.sum(F * wp ** 2) / np.sum(wp ** 4); CQ = np.sum(tq * wp ** 2) / np.sum(wp ** 4)
print(f'C_T = {CT:.3e} N/(rad/s)^2, C_Q = {CQ:.3e} N m/(rad/s)^2')
for p, f, t in zip(P, F, tq):
    print(f"  {p['tag']:>18s}  w {p['omega']:7.0f}  C_T {f / p['omega'] ** 2:.2e}  C_Q {t / p['omega'] ** 2:.2e}")


def coast(J, w0, tt, cq):
    # RK4 of J dw/dt = -(cq w^2 + Ke I_0(w)) / 運動方程式の 4 次ルンゲ・クッタ
    out = np.empty_like(tt); out[0] = w0
    f = lambda x: -(cq * x * x + Ke * (a0 + b0 * x * x)) / J
    for i in range(1, len(tt)):
        h = tt[i] - tt[i - 1]; x = out[i - 1]
        k1 = f(x); k2 = f(x + h / 2 * k1); k3 = f(x + h / 2 * k2); k4 = f(x + h * k3)
        out[i] = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


# J from coast runs (data use K_e = 5.905e-4 for w = v_m / K_e) / 惰性の各回から J
rows = {}
with open(D / 'coast_omega.csv') as fh:
    for r in csv.DictReader(fh):
        rows.setdefault((r['run'], r['config']), []).append((float(r['t_after_stop_s']), float(r['vm_V'])))
for (run, cfg), xy in rows.items():
    t = np.array([a for a, _ in xy]); om = np.array([b for _, b in xy]) / Ke
    s = (t > 0.005) & (om > 500); tt, ww = t[s], om[s]          # skip the first 5 ms (L and R*I drop) / 最初の 5 ms を除く
    cq = CQ if cfg == 'prop' else 0.0
    res = minimize_scalar(lambda J: np.mean((coast(J, ww[0], tt, cq) - ww) ** 2),
                          bounds=(1e-10, 1e-7), method='bounded', options={'xatol': 1e-12})
    print(f'  J {run:>20s} ({cfg:6s}) = {res.x:.3e} kg m^2   rms {100 * np.sqrt(res.fun) / ww.mean():.1f} %')
