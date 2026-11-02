# Three loads per duty (no prop, 4-blade, 3-blade): is V - K_e*omega a straight line in the current with a common
# intercept? Model: V - K_e*omega = R_out(D)*I + V_b. Each step uses its own in-step K_e.
# Usage: python3 tools/analyze_loads.py <data dir> (same layout as analyze_ke_probe.py)
# 各 Duty で負荷 3 通り（プロペラなし・4 枚・3 枚）: V - K_e·ω は電流に対して直線で、切片は共通か。
# モデル: V - K_e·ω = R_out(D)·I + V_b。各段は段の中で測った K_e を使う。
import io, contextlib, runpy, sys
import numpy as np

D = sys.argv[1]
SCRIPT = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'analyze_ke_probe.py')
sys.argv = ['x', D]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(SCRIPT)
deltas = g['deltas']
import os
extra = []
for sub in ('p3', 'p4t'):
    if os.path.exists(f'{D}/{sub}/points.csv'):
        extra += [s for s in g['run'](sub, deltas) if not (sub == 'p3' and s['lap'] == '1')]   # p3 lap 1 abnormal / p3 の 1 周目は異常
steps = g['P'] + g['NP'] + extra
steps = [s for s in steps if np.isfinite(s['ke']) and s['w_ok']]
for s in steps:
    s['y'] = s['V'] - s['ke'] * s['w']
print('load  lap duty  omega  I_m     V-K_e*omega   K_e')
for s in sorted(steps, key=lambda s: (s['duty'], s['sub'], s['lap'])):
    print(f"{s['sub']:3s}  {s['lap']}  D{s['duty']}  {s['w']:5.0f}  {s['I_m']:.4f}  {s['y']:.4f}       {s['ke']:.4e}")
duties = sorted({s['duty'] for s in steps})
print('\nper-duty line fit y = R*I + V_b (all loads and laps):')
for d in duties:
    ss = [s for s in steps if s['duty'] == d]
    I = np.array([s['I_m'] for s in ss]); y = np.array([s['y'] for s in ss])
    if len({s['sub'] for s in ss}) < 2:
        continue
    (R, vb), res, *_ = np.linalg.lstsq(np.c_[I, np.ones_like(I)], y, rcond=None)
    r = y - (R * I + vb)
    print(f"  D{d}: R_out {R:.3f} ohm, V_b {1000 * vb:+.0f} mV, rms resid {1000 * np.std(r):.1f} mV, loads {sorted({s['sub'] for s in ss})}")
# common V_b, one R per duty / V_b 共通、R は Duty ごと
rows, cols = [], []
idx = {d: i for i, d in enumerate(duties)}
A = np.zeros((len(steps), len(duties) + 1)); yv = np.zeros(len(steps))
for k, s in enumerate(steps):
    A[k, idx[s['duty']]] = s['I_m']; A[k, -1] = 1.0; yv[k] = s['y']
sol, *_ = np.linalg.lstsq(A, yv, rcond=None); res = yv - A @ sol
print(f"\ncommon V_b fit: V_b {1000 * sol[-1]:+.0f} mV, R_out " + ' '.join(f'D{d}:{sol[idx[d]]:.3f}' for d in duties)
      + f", rms resid {1000 * np.std(res):.1f} mV")
for load in ('np', 'p', 'p3'):
    rr = [res[k] for k, s in enumerate(steps) if s['sub'] == load]
    if rr:
        print(f'   residual mean for {load}: {1000 * np.mean(rr):+.1f} mV (n={len(rr)})')

# model comparison with equal parameter counts / パラメータ数をそろえた説の比較
def fit(extra_col):
    A = np.zeros((len(steps), len(duties) + 1)); yv = np.zeros(len(steps))
    for k, s in enumerate(steps):
        A[k, idx[s['duty']]] = s['I_m']; A[k, -1] = extra_col(s); yv[k] = s['y']
    sol, *_ = np.linalg.lstsq(A, yv, rcond=None); r = yv - A @ sol
    return sol, r
for name, col in (('model 1: constant drop V_b', lambda s: 1.0), ('model 2: c*omega*I', lambda s: s['w'] * s['I_m'])):
    sol, r = fit(col)
    print(f"\n{name}: common parameter {sol[-1]:.4e}, rms resid {1000 * np.std(r):.1f} mV")
    for load in ('np', 'p', 'p3', 'p4t'):
        rr = [r[k] for k, s in enumerate(steps) if s['sub'] == load]
        if rr:
            print(f"   {load:3s}: mean resid {1000 * np.mean(rr):+6.1f} mV, rms {1000 * np.sqrt(np.mean(np.square(rr))):5.1f} mV (n={len(rr)})")
