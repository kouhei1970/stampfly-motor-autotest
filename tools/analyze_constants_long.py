#!/usr/bin/env python3
"""Redo the 2026-10-15 constant fit (K_e, R, V_b, I_0(w), C_T, C_Q, J) from exp_long.py 'cycle' runs.
2026-10-15 の定数の当てはめ（K_e, R, V_b, I_0(w), C_T, C_Q, J）を、exp_long.py の 'cycle' 測定からやり直す。

Usage: nice -n 10 python3 tools/analyze_constants_long.py <D> --np <sub> --p <sub> [--coast coast] [--calib FILE] [--ke-ref 5.763e-4]
  <D>    holds scope_cal.csv (dithered contact checks) and the run subfolders (points.csv, events.csv, *_ke.npz, img/)
  --np   no-prop run subfolder (K_e, R, V_b, I_0(w), rotor J);  --p  with-prop run subfolder (C_T, C_Q, prop J)
  --coast  folder of coast_*.npz (relative to <D> if it exists there, else a path); *_np_* rotor only, *_p_* with propeller
Writes <D>/analysis_constants_steps.csv and <D>/analysis_constants.md (and scale_readings.csv in a --p folder lacking one).
"""
import argparse, csv, glob, json, os, re, sys
import numpy as np
from scipy.optimize import least_squares

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
G_N = 9.80665e-3                 # gram -> newton / g から N へ
REF = dict(Ke=5.90e-4, R=0.594, CT=7.60e-9, CQ=3.38e-11, a=0.044, b=2.9e-9, J_p=1.54e-8, J_r=3.46e-9)   # 2026-10-15
WARN = []                        # anything suspicious, printed and written / 気になった点


def load_probe_tools(D):
    # exec analyze_ke_probe.py up to its module-level run (it would read D/p and D/np itself); keep run(), delta_v_m()
    # analyze_ke_probe.py を、直下の実行部の手前まで exec する（D/p と D/np を勝手に読ませないため）
    src = open(os.path.join(TOOLS, 'analyze_ke_probe.py')).read().split('\ndeltas = delta_v_m()')[0]
    old = sys.argv; sys.argv = ['x', D]
    g = {'__file__': os.path.join(TOOLS, 'analyze_ke_probe.py'), '__name__': 'ke_probe'}
    exec(compile(src, 'analyze_ke_probe.py', 'exec'), g)
    sys.argv = old
    return g


def lsq(A, y):
    # least squares with residuals / 最小二乗と残差
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    return c, y - A @ c


def electrical_steps(g, D, sub, deltas, load):
    # per-step electrical data: run() of analyze_ke_probe + Vm_avg, step number and recorded omega from points.csv
    # 各段の電気量: analyze_ke_probe.run() に、points.csv の Vm_avg・段番号・記録された ω を足す
    out = g['run'](sub, deltas)
    pts = [r for r in csv.DictReader(open(f'{D}/{sub}/points.csv')) if r['skipped'] in ('0', '')]
    assert len(pts) == len(out)
    for s, r in zip(out, pts):
        assert int(r['duty_cmd']) == s['duty']
        dl = float(np.interp(s['t'], [x[0] for x in deltas], [x[1] for x in deltas]))
        s.update(load=load, step=int(r['step']), delta=dl, Vm=float(r['Vm_avg_V']) - dl, w_rec=float(r['omega_rad_s']),
                 I_raw=float(r['Ibat_A']), lap=int(s['lap']))
        s['w_research'] = abs(s['w'] / s['w_rec'] - 1) > 0.1          # recorded ripple line was wrong / 記録の脈動線が誤り
        s['y'] = s['Vm'] - s['ke'] * s['w']
        if s['w_research']:
            WARN.append(f"{sub} lap{s['lap']} D{s['duty']}: recorded omega {s['w_rec']:.0f} wrong, re-searched {s['w']:.0f}")
        if not np.isfinite(s['ke']):
            WARN.append(f"{sub} lap{s['lap']} D{s['duty']}: K_e probe gave no 10-ms coast window (n={s['n']}; capture too short after the stop?); no in-step K_e")
    return out


def scale_thrust(D, sub, calib, steps):
    # thrust per step: camera lag from the start/stop edges, zero windows between steps, early window before the K_e probe
    # 各段の推力: 起動・停止の立ち上がりからカメラ遅れ、段の間のゼロ窓、K_e プローブ前の窓
    import read_scale as RS, analyze_thrust as AT
    from pathlib import Path
    rd = Path(f'{D}/{sub}')
    if not (rd / 'scale_readings.csv').exists():
        RS.process_run(rd, json.load(open(calib)))
    s, ev, n_total = AT.load_run(rd)
    if len(s) < n_total * 0.5:
        WARN.append(f'{sub}: only {len(s)}/{n_total} scale frames readable')
    cmd = ev[ev.kind == 'cmd']
    starts = {(int(m.group(1)), int(m.group(2))): float(r.wall_time_s) for _, r in cmd.iterrows()
              if str(r.detail).startswith('motor test') and 'ke probe' not in str(r.note)
              and (m := re.search(r'lap(\d+) step(\d+)', str(r.note)))}
    pstop = {(int(m.group(1)), int(m.group(2))): float(r.wall_time_s) for _, r in cmd.iterrows()
             if (m := re.search(r'ke probe stop lap(\d+) step(\d+)', str(r.note)))}
    prest = {(int(m.group(1)), int(m.group(2))): float(r.wall_time_s) for _, r in cmd.iterrows()
             if 'ke probe restart' in str(r.note) and (m := re.search(r'lap(\d+) step(\d+)', str(r.note)))}
    final = {(int(m.group(1)), int(m.group(2))): float(r.wall_time_s) for _, r in cmd.iterrows()
             if (m := re.search(r'zero window after lap (\d+) step (\d+)', str(r.note)))}
    # camera lag: half-level crossings at every start and every final stop / カメラ遅れ: 起動と最終停止の半値交差
    dl, ks = [], sorted(starts)
    for k in ks:
        nxt = final.get(k, np.nan)
        d = AT.crossing_delay(s, starts[k], nxt, 'start'); dl += [d] if d is not None else []
        if k in final:
            later = [starts[j] for j in ks if starts[j] > final[k]]
            d = AT.crossing_delay(s, final[k], later[0] if later else np.nan, 'stop'); dl += [d] if d is not None else []
    L = float(np.median(dl)) if len(dl) >= 2 else AT.L_FALLBACK
    if len(dl) < 2:
        WARN.append(f'{sub}: camera lag not estimable, fallback {L} s')
    sd_L = float(np.std(dl)) if len(dl) > 1 else np.nan
    starts_all, stops_all = AT.parse_commands(ev)
    zp = AT.zero_periods(s, ev, starts_all, stops_all, L)
    zt, zv = np.array([z['t'] for z in zp]), np.array([z['zero'] for z in zp])
    scene, v = s.wall_time_s.values - L, s.value_g.values
    info = dict(L=L, sd_L=sd_L, n_L=len(dl), zeros=[z['zero'] for z in zp], n_frames=n_total, n_ok=len(s))
    for st in steps:
        k = (st['lap'], st['step'])
        if k not in starts:
            continue
        t0 = starts[k]; tf = final.get(k, np.nan); tp = pstop.get(k, tf); tr = prest.get(k, tp)
        z = lambda m: v[m] - np.interp(scene[m], zt, zv)
        me = (scene >= t0 + AT.WIN_START) & (scene <= tp - AT.WIN_END)
        ma = (scene >= t0 + AT.WIN_START) & (scene <= tf - AT.WIN_END) & ~((scene > tp - 0.3) & (scene < tr + 1.0))
        ml = (scene >= tr + 1.0) & (scene <= tf - AT.WIN_END)
        st.update(n_frames=int(me.sum()), zero_g=float(np.interp(scene[me].mean(), zt, zv)) if me.any() else np.nan,
                  F_g=float(z(me).mean()) if me.any() else np.nan, F_sd_g=float(z(me).std(ddof=1)) if me.sum() > 1 else np.nan,
                  F_all_g=float(z(ma).mean()) if ma.any() else np.nan, F_late_g=float(z(ml).mean()) if ml.any() else np.nan)
        if me.sum() < 4:
            WARN.append(f'{sub} lap{st["lap"]} D{st["duty"]}: only {int(me.sum())} readable frames in the thrust window')
    return info


def coast_trace(path, win=0.002, t_skip=0.005, w_min=500.0):
    # coast capture -> (t after the last ON pulse, v_m window means); delta from the final rest plateau when it is flat
    # 惰性波形 → 最後の ON パルス後の時刻と v_m の窓平均。停止後の定常値が平らならそれを v_m のずれにする
    d = np.load(path); t, v1, v2 = d['t'], d['v1'], d['v2']; fs = 1 / (t[1] - t[0]); k = int(round(win * fs))
    low = np.flatnonzero(v1 < 0.2)                                     # ON pulses pull the drain to ~0 V / ON は約 0 V
    t_stop = t[low[-1]] if low.size else np.nan
    n = len(t) // k; vm = (v2 - v1)[:n * k].reshape(n, k).mean(1); tc = t[:n * k].reshape(n, k).mean(1)
    tail = vm[-int(0.1 / win):]
    plateau = float(tail.mean()) if (tail.std() < 0.003 and abs(tail.mean()) < 0.03) else np.nan
    return tc - t_stop, vm, plateau, t_stop, int(low.size)


def scope_delta(folders):
    # (wall time, v_m offset) of the dithered checks in the first scope_cal.csv found; implausible rows (> 100 mV) dropped
    # 最初に見つかった scope_cal.csv のディザ確認の (時刻, v_m のずれ)。100 mV を超える行は除く
    for f in folders:
        p = os.path.join(f, 'scope_cal.csv')
        if os.path.exists(p):
            rows = [r for r in csv.DictReader(open(p)) if 'dither' in r['ch1_setting'] and abs(float(r['CH1_minus_CH2_mV'])) < 100]
            return [(float(r['wall_time_s']), -float(r['CH1_minus_CH2_mV']) / 1000) for r in rows]
    return [(0.0, 0.0)]


def coast_model(J, w0, tt, cq, ke, a, b):
    # RK4 of J dw/dt = -(cq w^2 + ke I_0(w)) on the (non-uniform) time nodes tt / 運動方程式の 4 次ルンゲ・クッタ
    f = lambda x: -(cq * x * x + ke * (a + b * x * x)) / J
    out = np.empty_like(tt); out[0] = w0
    for i in range(1, len(tt)):
        h = tt[i] - tt[i - 1]; x = out[i - 1]
        k1 = f(x); k2 = f(x + h / 2 * k1); k3 = f(x + h / 2 * k2); k4 = f(x + h * k3)
        out[i] = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def fit_J(tt, ww, cq, ke, a, b):
    # fit J and the initial omega to w(t) by relative residuals / 相対残差で J と初期 ω を当てはめる
    with np.errstate(all='ignore'):
        r = least_squares(lambda x: (coast_model(x[0] * 1e-8, x[1], tt, cq, ke, a, b) - ww) / ww, [1.5, ww[0]], x_scale=[1.0, ww[0]], bounds=([0.01, 500.0], [100.0, 3e4]))
    return r.x[0] * 1e-8, r.x[1], float(np.sqrt(np.mean(r.fun ** 2)) * 100)


def fit_shape(tt, ww):
    # diagnostic: free shape dw/dt = -(alpha + beta w^2) with free initial omega (no J, no I_0) / 診断: J も I_0 も使わない自由形状の当てはめ
    with np.errstate(all='ignore'):
        r = least_squares(lambda x: (coast_model(1.0, x[2], tt, x[1] * 1e-4, 1.0, x[0] * 1e4, 0.0) - ww) / ww, [1.0, 1.0, ww[0]],
                          x_scale=[1.0, 1.0, ww[0]], bounds=([-50, 0, 500.0], [50, 100, 3e4]))
    return r.x[0] * 1e4, r.x[1] * 1e-4, float(np.sqrt(np.mean(r.fun ** 2)) * 100)


def coast_all(files, d_list, ke_ref, ke_alt, a, b, cq):
    # J for every coast capture: omega = v_m / K_e, K_e = ke_ref (and ke_alt for comparison) / 惰性ごとの J
    rows = []
    for f in files:
        name = os.path.basename(f)[:-4]; rotor = '_np_' in name
        if not rotor and cq is None:
            WARN.append(f'{name}: with-prop coast skipped (no C_Q because no --p run)'); continue
        t, vm, plateau, t_stop, n_on = coast_trace(f)
        d_scope = float(np.interp(os.path.getmtime(f), [x[0] for x in d_list], [x[1] for x in d_list]))   # by file time / ファイル時刻で補間
        delta = plateau if np.isfinite(plateau) else d_scope
        if not np.isfinite(plateau):
            WARN.append(f'{name}: no flat rest plateau at the end, v_m offset taken from scope_cal ({1e3 * d_scope:.1f} mV)')
        v = vm - delta; sel = (t > 0.005) & (v > 500 * ke_ref)
        if sel.sum() < 20:
            WARN.append(f'{name}: only {int(sel.sum())} usable windows (stop not found?)'); continue
        tt, vv = t[sel], v[sel]
        rec = dict(name=name, kind='rotor' if rotor else 'prop', delta_mV=1e3 * delta, delta_scope_mV=1e3 * d_scope, t_stop=t_stop,
                   n_win=int(sel.sum()), w0=vv[0] / ke_ref)
        for tag, ke in (('ref', ke_ref), ('alt', ke_alt)):
            J, w0, rms = fit_J(tt, vv / ke, 0.0 if rotor else cq, ke, a, b)
            rec.update({f'J_{tag}': J, f'rms_{tag}': rms, f'w0fit_{tag}': w0})
        al, be, rec['rms_free'] = fit_shape(tt, vv / ke_ref)
        rec.update(alpha=al, beta=be, J_beta=((0.0 if rotor else cq) + ke_ref * b) / be if be > 0 else np.nan, J_alpha=ke_ref * a / al if al > 0 else np.nan)
        if rec['rms_ref'] > 3:
            WARN.append(f"{name}: coast fit rms {rec['rms_ref']:.1f} % (poor fit)")
        rows.append(rec)
    return rows


def per_duty_table(steps, load, keys):
    # lap-to-lap table: for each duty the first and last occurrence of each quantity and the difference / Duty ごとに初回と最終回の値と差
    lines = ['| duty | ' + ' | '.join(k for k, _, _ in keys) + ' |', '|' + '---|' * (len(keys) + 1)]
    diffs = {k: [] for k, _, _ in keys}
    for d in sorted({s['duty'] for s in steps if s['load'] == load}):
        ss = sorted([s for s in steps if s['load'] == load and s['duty'] == d], key=lambda s: (s['lap'], s['step']))
        cells = []
        for k, f, fmt in keys:
            x = [f(s) for s in ss if np.isfinite(f(s))]
            if len(x) >= 2:
                dp = 100 * (x[-1] / x[0] - 1); diffs[k].append(dp)
                cells.append(f'{fmt % x[0]} / {fmt % x[-1]} ({dp:+.1f}%{"" if len(x) == 2 else f", n={len(x)}"})')
            else:
                cells.append('-')
        lines.append(f'| {d} | ' + ' | '.join(cells) + ' |')
    lines.append('| rms diff [%] | ' + ' | '.join(f'{np.sqrt(np.mean(np.square(diffs[k]))):.1f}' if diffs[k] else '-' for k, _, _ in keys) + ' |')
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('D'); ap.add_argument('--np'); ap.add_argument('--p'); ap.add_argument('--coast', default='coast')
    ap.add_argument('--calib', default=os.path.join(TOOLS, 'scale_calib_2026-11-02b.json')); ap.add_argument('--ke-ref', type=float, default=5.763e-4)
    A = ap.parse_args(); D = A.D.rstrip('/')
    if not (A.np or A.p):
        sys.exit('give --np and/or --p')
    g = load_probe_tools(D); deltas = g['delta_v_m']()
    bad = [lab for _, x, lab in deltas if abs(x) > 0.1]
    deltas = [d for d in deltas if abs(d[1]) <= 0.1]       # a check with CH1 unplugged would spoil the interpolation / CH1 未接続の確認は除く
    WARN.extend(f'scope_cal row {lab} dropped (|offset| > 100 mV)' for lab in bad)
    steps = []
    if A.np:
        steps += electrical_steps(g, D, A.np, deltas, 'np')
    if A.p:
        steps += electrical_steps(g, D, A.p, deltas, 'p')
    for s in steps:
        s.update(F_g=np.nan, F_sd_g=np.nan, F_all_g=np.nan, F_late_g=np.nan, n_frames=0, zero_g=np.nan)
    sc = scale_thrust(D, A.p, A.calib, [s for s in steps if s['load'] == 'p']) if A.p else None
    md = []; P = md.append
    ok = [s for s in steps if np.isfinite(s['Vm']) and np.isfinite(s['w']) and np.isfinite(s['I_m'])]
    Ks = [s for s in ok if np.isfinite(s['ke'])]
    out = {}

    # 1. K_e and R (and V_b) / K_e と R（と V_b）
    I, w, Vm = (np.array([s[k] for s in ok]) for k in ('I_m', 'w', 'Vm'))
    (R1, Ke1), r1 = lsq(np.c_[I, w], Vm)
    (Vb3, R3, Ke3), r3 = lsq(np.c_[np.ones_like(I), I, w], Vm)
    fits = [('<v_m> = R I + K_e w (2 par)', f'{Ke1:.4e}', f'{R1:.3f}', '-', 1e3 * r1.std(), len(ok)),
            ('<v_m> = V_b + R I + K_e w (3 par)', f'{Ke3:.4e}', f'{R3:.3f}', f'{1e3 * Vb3:+.0f}', 1e3 * r3.std(), len(ok))]
    ke_step = {}
    for lab, sel in (('all', Ks), ('np', [s for s in Ks if s['load'] == 'np']), ('p', [s for s in Ks if s['load'] == 'p'])):
        if len(sel) >= 3 and (lab == 'all' or len(sel) > 2):
            Ii, yy = np.array([s['I_m'] for s in sel]), np.array([s['y'] for s in sel])
            (Rs, Vbs), rs = lsq(np.c_[Ii, np.ones_like(Ii)], yy)
            kes = np.median([s['ke'] for s in sel]); ke_step[lab] = kes
            fits.append((f'<v_m> - K_e,step w = R I + V_b ({lab})', f'{kes:.4e} (median step)', f'{Rs:.3f}', f'{1e3 * Vbs:+.0f}', 1e3 * rs.std(), len(sel)))
            out[f'R_step_{lab}'], out[f'Vb_step_{lab}'] = Rs, Vbs
    ke_alt = ke_step.get('all', A.ke_ref)

    # 2. I_0(w) from the no-prop steps / プロペラなしから無負荷電流
    npk = [s for s in ok if s['load'] == 'np']
    a0 = b0 = np.nan; rI = np.array([np.nan])
    if len(npk) >= 3:
        wn, In = np.array([s['w'] for s in npk]), np.array([s['I_m'] for s in npk])
        (a0, b0), rI = lsq(np.c_[np.ones_like(wn), wn ** 2], In)
        if a0 < 0 or wn.min() > 0.6 * wn.max():
            WARN.append(f'I_0 fit a={a0:.4f} A (negative or steps span only {wn.min():.0f}..{wn.max():.0f} rad/s): I_0 is extrapolated below that range, rotor J is unreliable')
    for s in steps:
        s['I0_fit'] = a0 + b0 * s['w'] ** 2 if np.isfinite(a0) else np.nan
        tq = s['ke'] * (s['I_m'] - s['I0_fit']) if s['load'] == 'p' else np.nan
        s['tq'] = tq; s['CQ'] = tq / s['w'] ** 2
        s['F_N'] = s['F_g'] * G_N; s['CT'] = s['F_N'] / s['w'] ** 2

    # 3. C_T three ways, C_Q / C_T の 3 通りと C_Q
    pk = [s for s in steps if s['load'] == 'p' and np.isfinite(s['F_N']) and np.isfinite(s['w'])]
    ct = {}
    if pk:
        wp, F = np.array([s['w'] for s in pk]), np.array([s['F_N'] for s in pk])
        ct['origin'] = np.sum(F * wp ** 2) / np.sum(wp ** 4); ct['origin_res'] = F - ct['origin'] * wp ** 2
        (c2, bo), ct['off_res'] = lsq(np.c_[wp ** 2, -np.ones_like(wp)], F); ct['off'], ct['b'] = c2, bo
        ct['step'] = F / wp ** 2
    cq = {}
    qk = [s for s in pk if np.isfinite(s['tq'])]
    if qk:
        wq, tq = np.array([s['w'] for s in qk]), np.array([s['tq'] for s in qk])
        cq['fit'] = np.sum(tq * wq ** 2) / np.sum(wq ** 4); cq['res'] = tq - cq['fit'] * wq ** 2
        cq['step'] = tq / wq ** 2
    elif A.p:
        WARN.append('no prop step has both thrust and K_e/I_0: C_Q not fitted')

    # 4. J from the coasts / 惰性から J
    cs = []; cdir = f'{D}/{A.coast}' if (not os.path.isabs(A.coast) and os.path.isdir(f'{D}/{A.coast}')) else A.coast
    files = sorted(glob.glob(f'{cdir}/*.npz')) if os.path.isdir(cdir) else []
    if files and np.isfinite(a0):
        cs = coast_all(files, scope_delta([cdir, os.path.dirname(os.path.abspath(cdir)), D]), A.ke_ref, ke_alt, a0, b0, cq.get('fit'))
    elif files:
        WARN.append('coast files found but no no-prop fit (I_0): J skipped')
    elif os.path.exists(A.coast) or os.path.exists(f'{D}/{A.coast}'):
        WARN.append(f'no *.npz in {cdir}')

    # ---- markdown ----
    P(f'# 定数の当てはめ（長時間測定 cycle） / constant fit from exp_long cycle runs\n\n`{D}`  np=`{A.np}`  p=`{A.p}`  coast=`{cdir if files else None}`  K_e(ref, coast omega)={A.ke_ref:.4e}\n')
    P(f'v_m offset (dithered checks) [mV]: ' + ', '.join(f'{lab} {1e3 * x:.1f}' for _, x, lab in deltas) + f'. 段ごとに時刻補間。steps: {len(steps)} (np {sum(s["load"] == "np" for s in steps)}, p {sum(s["load"] == "p" for s in steps)}), 使用 {len(ok)}、段内 K_e あり {len(Ks)}。\n')
    P('## 1. 定数表（2026-10-15 との比較） / constants vs 2026-10-15\n')
    P('| 量 | 今回 | 2026-10-15 | 備考 |\n|---|---|---|---|')
    P(f'| K_e (2 par, 全段) [V s/rad] | {Ke1:.4e} | {REF["Ke"]:.2e} | `<v_m> = R I + K_e w`、I は休止電流を引いた値 |')
    P(f'| R (2 par) [ohm] | {R1:.3f} | {REF["R"]:.3f} | |')
    P(f'| K_e / R / V_b (3 par) | {Ke3:.4e} / {R3:.3f} / {1e3 * Vb3:+.0f} mV | - | |')
    for lab in ('all', 'np', 'p'):
        if lab in ke_step:
            P(f'| 段内 K_e 中央値 ({lab}) | {ke_step[lab]:.4e} | - | n={sum(1 for s in Ks if lab == "all" or s["load"] == lab)}、範囲 {min(s["ke"] for s in Ks if lab == "all" or s["load"] == lab):.3e}..{max(s["ke"] for s in Ks if lab == "all" or s["load"] == lab):.3e} |')
            P(f'| R, V_b (段内 K_e, {lab}) | {out[f"R_step_{lab}"]:.3f} ohm, {1e3 * out[f"Vb_step_{lab}"]:+.0f} mV | - | `<v_m> - K_e,step w = R I + V_b` |')
    if np.isfinite(a0):
        P(f'| I_0(w) = a + b w^2 | {a0:.4f} + {b0:.2e} w^2 | {REF["a"]} + {REF["b"]:.1e} w^2 | プロペラなし {len(npk)} 段、残差 rms {1e3 * rI.std():.1f} mA |')
    if ct:
        P(f'| C_T 原点を通る [N/(rad/s)^2] | {ct["origin"]:.3e} | {REF["CT"]:.2e} | rms 残差 {1e3 * ct["origin_res"].std():.3f} mN |')
        P(f'| C_T 共通オフセット (F = C_T w^2 - b) | {ct["off"]:.3e}, b = {1e3 * ct["b"]:+.3f} mN | - | rms 残差 {1e3 * ct["off_res"].std():.3f} mN |')
        P(f'| C_T 段ごと（平均 ± 標準偏差） | {np.mean(ct["step"]):.3e} ± {np.std(ct["step"]):.1e} | - | 下の表を参照 |')
    if cq:
        P(f'| C_Q [N m/(rad/s)^2] | {cq["fit"]:.3e}（段ごと平均 {np.mean(cq["step"]):.3e}） | {REF["CQ"]:.2e} | `C_Q w^2 = K_e,step (I - I_0(w))`、rms 残差 {1e3 * cq["res"].std():.4f} mN m |')
    for kind, lab, ref in (('rotor', 'J ロータのみ', REF['J_r']), ('prop', 'J プロペラ付き', REF['J_p'])):
        c = [x for x in cs if x['kind'] == kind]
        if c:
            P(f'| {lab} [kg m^2] | {np.mean([x["J_ref"] for x in c]):.3e}（{len(c)} 回、範囲 {min(x["J_ref"] for x in c):.2e}..{max(x["J_ref"] for x in c):.2e}） | {ref:.2e} | K_e={A.ke_ref:.3e}。K_e={ke_alt:.3e}（段内中央値）なら {np.mean([x["J_alt"] for x in c]):.3e} |')
    if sc:
        P(f'| カメラ遅れ L | {sc["L"]:.2f} s（標準偏差 {sc["sd_L"]:.2f}、n={sc["n_L"]}） | - | 秤ゼロ点 {", ".join(f"{z:.2f}" for z in sc["zeros"][:12])} g、読めたコマ {sc["n_ok"]}/{sc["n_frames"]} |')
    P('\n## 2. 当てはめの内訳 / fits\n')
    P('| 当てはめ | K_e | R [ohm] | V_b [mV] | rms 残差 [mV] | n |\n|---|---|---|---|---|---|')
    for f in fits:
        P(f'| {f[0]} | {f[1]} | {f[2]} | {f[3]} | {f[4]:.1f} | {f[5]} |')
    if pk:
        P('\n| 段 | duty | w [rad/s] | F [mN] | C_T 段ごと | 原点fit残差 [mN] | オフセットfit残差 [mN] | C_Q 段ごと | K_e,step | F 前半/全体/後半 [g] |\n|---|---|---|---|---|---|---|---|---|---|')
        for i, s in enumerate(pk):
            q = s['CQ']
            P(f"| lap{s['lap']} s{s['step']} | {s['duty']} | {s['w']:.0f} | {1e3 * s['F_N']:.2f} | {s['CT']:.3e} | {1e3 * ct['origin_res'][i]:+.2f} | {1e3 * ct['off_res'][i]:+.2f} | {q:.3e} | {s['ke']:.3e} | {s['F_g']:.2f} / {s['F_all_g']:.2f} / {s['F_late_g']:.2f} |")
    P('\n## 3. 周回間のくり返し / repeatability (first / last occurrence of each duty, difference)\n')
    for load in ('np', 'p'):
        if any(s['load'] == load for s in steps):
            keys = [('w [rad/s]', lambda s: s['w'], '%.0f'), ('I_m [A]', lambda s: s['I_m'], '%.4f'), ('K_e', lambda s: s['ke'], '%.3e'),
                    ('V - K_e w [V]', lambda s: s['y'], '%.4f')]
            if load == 'p':
                keys += [('F [g]', lambda s: s['F_g'], '%.2f'), ('C_T', lambda s: s['CT'], '%.3e'), ('C_Q', lambda s: s['CQ'], '%.3e')]
            P(f'### {load}\n'); md.extend(per_duty_table(steps, load, keys)); P('')
    if cs:
        P('## 4. 惰性（スピンダウン） / coasts\n')
        P('| 取り込み | 種別 | 停止時刻 [s] | 窓数 | omega0 [rad/s] | v_m オフセット [mV]（定常/scope_cal） | J (K_e ref) | rms [%] | J (K_e 段内) | rms [%] |\n|---|---|---|---|---|---|---|---|---|---|')
        for x in cs:
            P(f"| {x['name']} | {x['kind']} | {x['t_stop']:.3f} | {x['n_win']} | {x['w0']:.0f} | {x['delta_mV']:.1f} / {x['delta_scope_mV']:.1f} | {x['J_ref']:.3e} | {x['rms_ref']:.2f} | {x['J_alt']:.3e} | {x['rms_alt']:.2f} |")
        P('診断: 形状自由の当てはめ dw/dt = -(alpha + beta w^2)（J・I_0 を使わない）。I_0 の定常値と整合するなら J_beta ≈ J_alpha ≈ J。\n')
        P('| 取り込み | alpha [rad/s^2] | beta [1/s] | rms 自由形状 [%] | J = (C_Q + K_e b) / beta | J = K_e a / alpha |\n|---|---|---|---|---|---|')
        for x in cs:
            P(f"| {x['name']} | {x['alpha']:.0f} | {x['beta']:.3e} | {x['rms_free']:.2f} | {x['J_beta']:.3e} | {x['J_alpha']:.3e} |")
        P('')
    P('## 5. 要注意 / warnings\n')
    P('\n'.join(f'- {w_}' for w_ in WARN) if WARN else '- なし')
    open(f'{D}/analysis_constants.md', 'w').write('\n'.join(md) + '\n')

    # per-step CSV / 段ごとの CSV
    cols = ['load', 'sub', 'lap', 'step', 'duty', 't', 'V', 'Vm', 'delta', 'w', 'w_rec', 'w_research', 'I_m', 'I_raw', 'ke', 'n', 'spread', 'gap', 'y', 'I0_fit', 'tq', 'CQ',
            'F_g', 'F_sd_g', 'F_all_g', 'F_late_g', 'n_frames', 'zero_g', 'F_N', 'CT']
    with open(f'{D}/analysis_constants_steps.csv', 'w', newline='') as f:
        wr = csv.writer(f); wr.writerow(cols); wr.writerows([[s.get(c, '') for c in cols] for s in steps])
    print('\n'.join(md))


if __name__ == '__main__':
    main()
