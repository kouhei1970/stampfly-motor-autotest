#!/usr/bin/env python3
"""Per-step rotor speed (anchored peak search) and drive-circuit resistance R_out for bench runs.
各定常段の回転数（独立な推定で探索範囲を絞ったピーク探索）と駆動回路の等価抵抗 R_out を求める。

Why an anchor: the largest FFT line of the per-PWM-period mean of V_DD - v_D is sometimes not the
commutation ripple (seen at duty 80 %).  The mean motor voltage gives E = Vm - R*I_m, so
omega_est = (Vm - R*I_m)/K_e is an independent estimate; the ripple line is searched only within
+-10 % of 10*omega_est/(2*pi) and then refined with fine_peak (analyze_rpm_sweep).
最大の FFT 線が脈動でないことがある（Duty 80 % 付近）。平均端子電圧から逆起電力 E = Vm - R*I_m を出し、
omega_est = E/K_e の ±10 % の範囲だけでピークを探してから fine_peak で細かく読む。

Usage:
  python3 tools/analyze_long.py steps <run folder> [<run folder> ...] [--ke VALUE]
      -> <run>/analysis_steps.csv per folder (exp_long run folder, or exp_auto folder with results.json)
  python3 tools/analyze_long.py summary [--data DIR] [--day 2026-11-02] [--out FILE]
      -> tables printed and written to <data>/<day>_summary_electrical.md
Folders: exp_long run = points.csv + events.csv + <tag>_L<lap>_S<step>.npz ; exp_auto = results.json + pwm_*.npz.
"""
import argparse, csv, json, pathlib, re, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import exp_rpm as E
import analyze_rpm_sweep as S    # sets E.F_PWM = 150096.67 Hz / PWM 周波数を実測値に設定

N_RIPPLE = 10            # ripples per revolution / 1 回転あたりの脈動の数
R_ANCHOR = 0.594         # ohm, used only for the anchor (E = Vm - R*I_m) / アンカー用の抵抗 [ohm]
WINDOW = 0.10            # +-10 % search window / 探索範囲
KE_DEFAULT = 5.759e-4    # fallback only; normally computed from the spin-down / 通常はスピンダウンから算出
KE_BENCH = 5.905e-4      # 2026-10-15 value / 2026-10-15 の値
POWER_RE = re.compile(r'([\d.]+)\s*V\s+(\d+)\s*mA')
COAST_FILE = 'coast_np_d60_ke.npz'


# ---------------------------------------------------------------- K_e from the spin-down
def ke_from_coast(path, win_s=0.020, vmin=0.5, fmin=600.0, fmax=12e3):
    # K_e = v_m/omega in consecutive 20-ms windows after the last on-pulse (current 0)
    # 最後のオンパルス後の 20 ms 窓ごとに v_m/omega（電流 0）
    d = np.load(path); t, v1, v2 = d['t'], d['v1'], d['v2']
    fs = 1 / (t[1] - t[0]); vm = v2 - v1
    last_on = np.flatnonzero(v1 < 0.5)[-1]
    n = int(win_s * fs); rows = []
    a = last_on + 1000
    while a + n <= t.size:
        x = vm[a:a + n]; tt = np.arange(n) / fs
        if x.mean() < vmin:
            break
        r = x - np.polyval(np.polyfit(tt, x, 3), tt)          # remove the slow decay / 緩やかな減衰を除く
        pad = 8; Fz = np.abs(np.fft.rfft(r * np.hanning(n), n * pad)); f = np.fft.rfftfreq(n * pad, 1 / fs)
        m = np.flatnonzero((f > fmin) & (f < fmax)); k = m[np.argmax(Fz[m])]
        la, lb, lc = np.log(Fz[k - 1:k + 2])
        fpk = f[k] + 0.5 * (la - lc) / (la - 2 * lb + lc) * (f[1] - f[0])
        rows.append((float(x.mean()), 2 * np.pi * float(fpk) / N_RIPPLE))
        a += n
    vmm = np.array([r[0] for r in rows]); om = np.array([r[1] for r in rows])
    return dict(median=float(np.median(vmm / om)), fit=float(np.sum(vmm * om) / np.sum(om ** 2)),
                n=len(rows), omega_min=float(om.min()), omega_max=float(om.max()),
                ratios=(vmm / om).tolist(), omegas=om.tolist(), vms=vmm.tolist())


# ---------------------------------------------------------------- one capture
def coarse_peak(avg):
    f, F = E.spectrum(avg, E.F_PWM)     # 150 Hz - 12 kHz, same as the original pipeline / 従来と同じ範囲
    return float(f[np.argmax(F)])


def analyse_capture(npz, im_a, ke):
    d = np.load(npz); avg, vdd, vm = S.per_period_mean(d['t'], d['v1'], d['v2'])
    f_old = S.fine_peak(avg, E.F_PWM, coarse_peak(avg))
    om_anchor = (vm - R_ANCHOR * im_a) / ke
    f_anchor = N_RIPPLE * om_anchor / (2 * np.pi)
    f_new = S.fine_peak(avg, E.F_PWM, f_anchor, half=WINDOW * f_anchor) if f_anchor > 0 else float('nan')
    return dict(VDD=vdd, Vm=vm, f_old=f_old, f_new=f_new, f_anchor=f_anchor,
                omega_old=2 * np.pi * f_old / N_RIPPLE, omega_new=2 * np.pi * f_new / N_RIPPLE, omega_anchor=om_anchor)


def parse_power(s):
    m = POWER_RE.search(s)
    return (float(m.group(1)), float(m.group(2))) if m else (float('nan'), float('nan'))   # V, mA


def finish(row, ke):
    # derived electrical quantities from omega_new / omega_new から電気量を計算
    im = row['I_motor_A']
    row['R_out'] = (row['VDD'] - ke * row['omega_new']) / im
    row['R_w'] = (row['Vm'] - ke * row['omega_new']) / im
    row['changed'] = int(abs(row['omega_new'] / row['omega_old'] - 1) > 1e-3)
    row['anchor_dev_new_pct'] = 100 * (row['omega_new'] / row['omega_anchor'] - 1)
    row['anchor_dev_old_pct'] = 100 * (row['omega_old'] / row['omega_anchor'] - 1)
    row['old_in_window'] = int(abs(row['omega_old'] / row['omega_anchor'] - 1) <= WINDOW)
    return row


# ---------------------------------------------------------------- run folders
def steps_long(folder, ke):
    pts = list(csv.DictReader(open(folder / 'points.csv')))
    ev = list(csv.DictReader(open(folder / 'events.csv')))
    run = json.loads((folder / 'run.json').read_text()) if (folder / 'run.json').exists() else {}
    n_motors = 4 if run.get('args', {}).get('all_motors') else 1
    rests = sorted((float(e['wall_time_s']), parse_power(e['detail'])[1]) for e in ev if e['kind'] == 'rest_power')
    out = []
    for p in pts:
        if int(p['skipped']) or not (folder / p['npz']).exists():
            continue
        t_ref = float(p['trigger_time'])
        before = [r for r in rests if r[0] <= t_ref + 0.5]
        rest_ma = (before[-1] if before else min(rests, key=lambda r: abs(r[0] - t_ref)))[1]
        raw_ma = parse_power(p['power_line'])[1]
        im = (raw_ma - rest_ma) / 1000 / n_motors
        r = analyse_capture(folder / p['npz'], im, ke)
        row = dict(run=folder.name, tag=p['npz'][:-4], section=p['section'], lap=int(p['lap']), step=int(p['step']),
                   duty_cmd=float(p['duty_cmd']) / 100, n_motors=n_motors, I_raw_mA=raw_ma, I_rest_mA=rest_ma,
                   I_motor_A=im, CLI_V=float(p['CLI_V']), omega_recorded=float(p['omega_rad_s']), **r)
        row['CLI_minus_VDD'] = row['CLI_V'] - row['VDD']
        row['old_matches_recorded'] = int(abs(row['omega_old'] / row['omega_recorded'] - 1) < 2e-3)
        out.append(finish(row, ke))
    return out


def steps_auto(folder, ke):
    res = json.loads((folder / 'results.json').read_text()); out = []
    for tag, r in res.items():
        if not tag.startswith('pwm_') or not (folder / f'{tag}.npz').exists():
            continue
        spin = [parse_power(s)[1] for s in r['power_spin_samples']]
        rest = np.mean([parse_power(r['power_before'])[1], parse_power(r['power_after'])[1]])
        im = (float(np.mean(spin)) - rest) / 1000
        c = analyse_capture(folder / f'{tag}.npz', im, ke)
        row = dict(run=folder.name, tag=tag, section='np' if '_np' in tag else 'prop', lap=2 if tag.endswith('_rep') else 1,
                   step=0, duty_cmd=r['duty_cmd'], n_motors=1, I_raw_mA=float(np.mean(spin)), I_rest_mA=float(rest),
                   I_motor_A=im, CLI_V=float('nan'), omega_recorded=float('nan'), **c)
        row['CLI_minus_VDD'] = float('nan'); row['old_matches_recorded'] = -1
        out.append(finish(row, ke))
    return out


COLUMNS = ['run', 'tag', 'section', 'lap', 'step', 'duty_cmd', 'n_motors', 'VDD', 'Vm', 'I_raw_mA', 'I_rest_mA', 'I_motor_A',
           'f_old', 'f_new', 'f_anchor', 'omega_old', 'omega_new', 'omega_anchor', 'omega_recorded', 'changed',
           'old_in_window', 'old_matches_recorded', 'anchor_dev_old_pct', 'anchor_dev_new_pct',
           'R_out', 'R_w', 'CLI_V', 'CLI_minus_VDD']


def run_folder(folder, ke, write=True):
    folder = pathlib.Path(folder)
    rows = steps_long(folder, ke) if (folder / 'points.csv').exists() else steps_auto(folder, ke)
    if write:
        with open(folder / 'analysis_steps.csv', 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(COLUMNS)
            for r in rows:
                w.writerow([f'{r[c]:.6g}' if isinstance(r[c], float) else r[c] for c in COLUMNS])
    return rows


# ---------------------------------------------------------------- summary
class Report:
    def __init__(self): self.lines = []
    def p(self, s=''): self.lines.append(s); print(s)
    def table(self, head, rows):
        self.p('| ' + ' | '.join(head) + ' |'); self.p('|' + '---|' * len(head))
        for r in rows:
            self.p('| ' + ' | '.join(r) + ' |')
        self.p()


def fx(v, n=3):
    return 'nan' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:.{n}f}'


def rout_with(r, ke):
    return (r['VDD'] - ke * r['omega_new']) / r['I_motor_A']


def rw_with(r, ke):
    return (r['Vm'] - ke * r['omega_new']) / r['I_motor_A']


def summary(data, day, out, ke_arg):
    data = pathlib.Path(data); rep = Report()
    b1, b2, b3 = data / f'{day}_bat1', data / f'{day}_bat2', data / f'{day}_bat3'
    ke_info = ke_from_coast(b1 / COAST_FILE)
    ke = ke_arg or ke_info['median']; ke_fit = ke_info['fit']
    rep.p(f'# Electrical summary {day}'); rep.p()
    rep.p('## 0. K_e and method'); rep.p()
    rep.p(f'- K_e from no-prop spin-down ({COAST_FILE}): {ke_info["n"]} windows of 20 ms, omega {ke_info["omega_min"]:.0f}-{ke_info["omega_max"]:.0f} rad/s '
          f'(window mean v_m > 0.5 V); median v_m/omega = {ke_info["median"]:.4e}, through-origin fit = {ke_fit:.4e} V s/rad; used for tables: {ke:.4e}')
    rep.p(f'- Reference 2026-10-15 value {KE_BENCH:.3e}. Anchor R = {R_ANCHOR} ohm, search window +-{WINDOW*100:.0f} %, N = {N_RIPPLE}, PWM {E.F_PWM:.2f} Hz.')
    rep.p('- I_m = reading - rest current (exp_auto: mean of power_before/after; exp_long: last rest_power before the step). Four-motor runs: motor part divided by 4 (equal split assumed).')
    rep.p('- R_out = (V_DD - K_e*omega)/I_m ; R_w = (Vm - K_e*omega)/I_m ; V_DD, Vm = scope means of each capture. R_w cannot be separated from a temperature change of K_e.')
    rep.p()
    runs = {'bat1': run_folder(b1, ke)}
    names2 = ['A1', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']; names3 = ['A4M', 'B4M', 'C4M']
    for n in names2: runs[n] = run_folder(b2 / n, ke)
    for n in names3: runs[n] = run_folder(b3 / n, ke)

    # --- omega check
    rep.p('### omega peak check (old = largest line, new = anchored)'); rep.p()
    allrows = [r for k in runs for r in runs[k]]
    ok = [r for r in allrows if r['old_in_window']]
    dev = max(abs(r['omega_new'] / r['omega_old'] - 1) for r in ok) * 100
    chg = [r for r in allrows if r['changed']]
    rep.p(f'- steps: {len(allrows)}; old value inside the +-10 % window of the anchor: {len(ok)}; max |new/old - 1| over those: {dev:.4f} %; '
          f'changed (>0.1 %): {len(chg)}; old not matching recorded omega in points.csv: '
          f'{sum(1 for r in allrows if r["old_matches_recorded"] == 0)}')
    rep.p(f'- max |new - anchor| over all steps: {max(abs(r["anchor_dev_new_pct"]) for r in allrows):.1f} % (large values mean the anchor is itself off)')
    rep.p()
    rep.table(['run', 'tag', 'duty', 'omega old', 'omega new', 'omega anchor', 'old dev from anchor %', 'new dev %'],
              [[r['run'], r['tag'], f'{r["duty_cmd"]:.2f}', fx(r['omega_old'], 1), fx(r['omega_new'], 1), fx(r['omega_anchor'], 1),
                fx(r['anchor_dev_old_pct'], 1), fx(r['anchor_dev_new_pct'], 1)] for r in chg])

    # --- (a) E measurement
    rep.p('## a. Measurement E: no-prop vs prop (exp_auto, bat1)'); rep.p()
    bat1 = runs['bat1']; duties = [0.2, 0.3, 0.4, 0.5, 0.6]
    def pick(sec, lap, duty):
        return next(r for r in bat1 if r['section'] == sec and r['lap'] == lap and abs(r['duty_cmd'] - duty) < 1e-6)
    for label, k in (('K_e median', ke), ('K_e fit', ke_fit), ('K_e 5.905e-4', KE_BENCH)):
        rows = []
        for d in duties:
            n1, n2, pr = pick('np', 1, d), pick('np', 2, d), pick('prop', 1, d)
            a, b, c = (rout_with(x, k) for x in (n1, n2, pr)); m = (a + b) / 2
            rows.append([f'{d:.1f}', fx(n1['omega_new'], 0), fx(n2['omega_new'], 0), fx(pr['omega_new'], 0),
                         fx(n1['VDD'], 3), fx(pr['VDD'], 3), fx(a), fx(b), fx(m), fx(c), fx(c / m)])
        rep.p(f'### {label} = {k:.4e}'); rep.p()
        rep.table(['duty', 'omega np', 'omega np rep', 'omega prop', 'VDD np', 'VDD prop', 'R_out np', 'R_out np rep', 'R_out np mean',
                   'R_out prop', 'prop / np mean'], rows)

    # --- (b) cold laps
    rep.p('## b. Cold laps A1, C1, C2, C3 (one motor, prop)'); rep.p()
    cold = ['A1', 'C1', 'C2', 'C3']
    rep.table(['lap', 'V_DD min', 'V_DD max'], [[n, fx(min(r['VDD'] for r in runs[n]), 3), fx(max(r['VDD'] for r in runs[n]), 3)] for n in cold])
    dl = sorted({round(r['duty_cmd'], 2) for r in runs['A1']})
    rows = []; rows_v = []
    for d in dl:
        rs = [next(r for r in runs[n] if abs(r['duty_cmd'] - d) < 1e-6) for n in cold]
        R = np.array([rout_with(r, ke) for r in rs]); V = np.array([r['VDD'] for r in rs])
        slope = np.polyfit(V, R, 1)[0] / R.mean() * 100
        rows.append([f'{d:.2f}'] + [fx(x) for x in R] + [fx(100 * (R[3] / R[0] - 1), 1), fx(slope, 1)])
        rows_v.append([f'{d:.2f}'] + [fx(x, 3) for x in V])
    rep.table(['duty', 'R_out A1', 'C1', 'C2', 'C3', 'C3/A1 change %', 'slope %/V (4-lap line fit)'], rows)
    rep.p('V_DD [V] per duty and lap:'); rep.p()
    rep.table(['duty'] + cold, rows_v)

    # --- (c) warm laps
    rep.p('## c. Warm laps B1-B3 (4 laps each, one motor)'); rep.p()
    labels = [(n, l) for n in ['B1', 'B2', 'B3'] for l in (1, 2, 3, 4)]
    for qname, fn in (('R_out', rout_with), ('R_w', rw_with)):
        rows = []
        for d in dl:
            vals = [fn(next(r for r in runs[n] if r['lap'] == l and abs(r['duty_cmd'] - d) < 1e-6), ke) for n, l in labels]
            ch = [100 * (vals[3 + 4 * i] / vals[4 * i] - 1) for i in range(3)]
            rows.append([f'{d:.2f}'] + [fx(v) for v in vals] + [fx(c, 1) for c in ch] + [fx(100 * (vals[-1] / vals[0] - 1), 1)])
        rep.p(f'### {qname} per lap, change first to last lap (%)'); rep.p()
        rep.table(['duty'] + [f'{n}L{l}' for n, l in labels] + ['B1 L1->L4 %', 'B2 L1->L4 %', 'B3 L1->L4 %', 'B1L1->B3L4 %'], rows)
    rep.table(['lap', 'V_DD mean'], [[f'{n}L{l}', fx(np.mean([r['VDD'] for r in runs[n] if r['lap'] == l]), 3)] for n, l in labels])

    # --- (d) four motors
    rep.p('## d. Four-motor runs (scope on motor 1, current = total of four; equal split assumed)'); rep.p()
    for n in names3:
        rep.p(f'### {n}'); rep.p()
        rep.table(['lap', 'duty', 'I_total reading A', 'I_rest mA', 'I_m1 A (equal split)', 'V_DD', 'CLI_V', 'CLI_V - V_DD', 'R_out m1'],
                  [[str(r['lap']), f'{r["duty_cmd"]:.2f}', fx(r['I_raw_mA'] / 1000), fx(r['I_rest_mA'], 0), fx(r['I_motor_A']), fx(r['VDD'], 4),
                    fx(r['CLI_V'], 3), fx(r['CLI_minus_VDD'], 3), fx(rout_with(r, ke))] for r in runs[n]])
    rep.p('### Linear fits of CLI_V - V_DD (offset in V)'); rep.p()
    rows = []
    for name, rs in (('A4M', runs['A4M']), ('B4M', runs['B4M']), ('C4M', runs['C4M']), ('4M all', runs['A4M'] + runs['B4M'] + runs['C4M']),
                     ('1M A1,B1-3,C1-3', [r for n in names2 for r in runs[n]])):
        off = np.array([r['CLI_minus_VDD'] for r in rs]); itot = np.array([r['I_raw_mA'] / 1000 for r in rs]); v = np.array([r['VDD'] for r in rs])
        pi = np.polyfit(itot, off, 1); pv = np.polyfit(v, off, 1)
        res_i = off - np.polyval(pi, itot)
        rows.append([name, str(len(rs)), fx(off.mean(), 3), fx(pi[0], 4), fx(pi[1], 3), fx(res_i.std(), 3), fx(pv[0], 3), fx(pv[1], 3)])
    rep.table(['set', 'n', 'mean offset V', 'slope vs I_total V/A (ohm)', 'intercept V', 'resid sd V', 'slope vs V_DD V/V', 'intercept V'], rows)
    pathlib.Path(out).write_text('\n'.join(rep.lines) + '\n')
    return ke_info


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('steps'); s.add_argument('folders', nargs='+'); s.add_argument('--ke', type=float)
    s.add_argument('--coast', help='spin-down npz for K_e (default: <day>_bat1/coast_np_d60_ke.npz next to the run) / K_e 用スピンダウン')
    m = sub.add_parser('summary'); m.add_argument('--data', default='data'); m.add_argument('--day', default='2026-11-02')
    m.add_argument('--out'); m.add_argument('--ke', type=float)
    a = ap.parse_args()
    if a.cmd == 'steps':
        ke = a.ke
        if ke is None:
            coast = pathlib.Path(a.coast) if a.coast else None
            if coast is None:
                for f in a.folders:
                    cand = list(pathlib.Path(f).resolve().parent.glob(f'*bat1/{COAST_FILE}')) + list(pathlib.Path(f).resolve().glob(COAST_FILE))
                    if cand: coast = cand[0]; break
            ke = ke_from_coast(coast)['median'] if coast else KE_DEFAULT
        for f in a.folders:
            rows = run_folder(f, ke)
            print(f'{f}: {len(rows)} steps, K_e={ke:.4e}, changed {sum(r["changed"] for r in rows)}')
    else:
        out = a.out or str(pathlib.Path(a.data) / f'{a.day}_summary_electrical.md')
        summary(a.data, a.day, out, a.ke)


if __name__ == '__main__':
    main()
