# R_out with an in-step K_e for every step (2026-11-02 evening run) and the no-prop/prop ratio test.
# 各段の中で測った K_e を使う R_out と、プロペラなし／付きの比の判定（2026-11-02 夕方の測定）。
# Usage: python3 tools/analyze_ke_probe.py <data dir with p/ and np/ (and p3/, p4t/) and scope_cal.csv>
import csv, re, sys
import numpy as np

D = sys.argv[1]
N = 10                                  # ripples per revolution / 1 回転あたりの脈動
R_W = 0.594                             # winding resistance for the omega anchor only / ω の見当用の巻線抵抗


def delta_v_m():
    # offset of v_m = CH2 - CH1 from the dithered same-condition checks: v_m reads -(CH1 - CH2) too high
    # オフセットを揺らした確認から v_m のずれ: v_m は -(CH1 - CH2) だけ高く出る
    rows = [r for r in csv.DictReader(open(f'{D}/scope_cal.csv')) if 'dither' in r['ch1_setting']]
    return [(float(r['wall_time_s']), -float(r['CH1_minus_CH2_mV']) / 1000, r['label']) for r in rows]


def ke_from_probe(path, delta, w_anchor):
    # coast = longest stretch without on-pulses; 10-ms windows from 3 ms after the stop; ripple frequency
    # searched within +-25 % of the window's expected value (starts at the steady omega and falls)
    # 惰性 = ON パルスのない最長区間。止めて 3 ms 後から 10 ms 窓。脈動の周波数は見当の ±25 % で探す
    d = np.load(path); t, v1, v2 = d['t'], d['v1'], d['v2']; fs = 1 / (t[1] - t[0])
    low = np.flatnonzero(v1 < 0.2)        # on-pulses pull the drain to ~0 V; the coast stays >= 0.4 V / ON は約 0 V、惰性は 0.4 V 以上
    if low.size < 2:
        return np.nan, 0, np.nan, np.nan
    i = int(np.argmax(np.diff(low))); t_a, t_b = t[low[i]], t[low[i + 1]]
    vm = v2 - v1; ks = []; w_prev = w_anchor
    for w0 in np.arange(t_a + 0.003, t_b - 0.011, 0.010):
        m = (t >= w0) & (t < w0 + 0.010); x = vm[m]
        if x.size < 1000:
            continue
        xx = x - x.mean(); F = np.abs(np.fft.rfft(xx * np.hanning(xx.size), 32 * xx.size))
        fr = np.fft.rfftfreq(32 * xx.size, 1 / fs); f0 = w_prev * N / (2 * np.pi)
        sel = (fr > 0.75 * f0) & (fr < 1.1 * f0)
        if not sel.any():
            continue
        w = 2 * np.pi * fr[sel][np.argmax(F[sel])] / N; w_prev = w
        ks.append((x.mean() - delta) / w)
    ks = np.array(ks)
    return (float(np.median(ks)) if ks.size else np.nan), int(ks.size), (float(np.ptp(ks) / np.median(ks) * 100) if ks.size else np.nan), 1e3 * (t_b - t_a)


def run(sub, deltas):
    rows = list(csv.DictReader(open(f'{D}/{sub}/points.csv')))
    ev = list(csv.DictReader(open(f'{D}/{sub}/events.csv')))
    rest = [(float(e['wall_time_s']), int(re.search(r'(-?\d+) mA', e['detail']).group(1)) / 1000)
            for e in ev if e['kind'] == 'rest_power' and 'mA' in e['detail']]
    out = []
    for r in rows:
        if r['skipped'] not in ('0', ''):
            continue
        t = float(r['trigger_time']); V = float(r['VDD_V']); I = float(r['Ibat_A']); Vm = float(r['Vm_avg_V'])
        i0 = min(rest, key=lambda x: abs(x[0] - t))[1]
        dl = float(np.interp(t, [x[0] for x in deltas], [x[1] for x in deltas]))
        # steady omega: check the recorded line against the mean-voltage anchor, else search near the anchor later
        # 定常の ω: 記録値を平均電圧からの見当と照合
        w = float(r['omega_rad_s']); w_est = (Vm - dl - R_W * (I - i0)) / 5.7e-4
        ok_w = abs(w / w_est - 1) < 0.1
        if not ok_w:
            # re-search the ripple line within +-10 % of the anchor in the steady capture / 定常の取り込みで見当の ±10 % を探し直す
            sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
            import analyze_rpm_sweep as RS, exp_rpm as E
            E.F_PWM = 150096.67
            dd = np.load(f"{D}/{sub}/{r['npz']}"); avg, _, _ = RS.per_period_mean(dd['t'], dd['v1'], dd['v2'])
            x = avg - avg.mean(); F = np.abs(np.fft.rfft(x * np.hanning(x.size), 16 * x.size)); fr = np.fft.rfftfreq(16 * x.size, 1 / E.F_PWM)
            f0 = w_est * N / (2 * np.pi); sel = (fr > 0.9 * f0) & (fr < 1.1 * f0)
            w = float(2 * np.pi * RS.fine_peak(avg, E.F_PWM, fr[sel][np.argmax(F[sel])]) / N); ok_w = True
        probe = f"{D}/{sub}/{r['npz'].replace('.npz', '_ke.npz')}"
        try:
            ke, n, spread, gap = ke_from_probe(probe, dl, w if ok_w else w_est)
        except FileNotFoundError:
            ke, n, spread, gap = np.nan, 0, np.nan, np.nan
        out.append(dict(sub=sub, lap=r['lap'], duty=int(r['duty_cmd']), t=t, V=V, w=w, w_ok=ok_w, I_m=I - i0,
                        ke=ke, n=n, spread=spread, gap=gap, R=(V - ke * w) / (I - i0) if ok_w else np.nan))
    return out


deltas = delta_v_m()
print('v_m offset from dithered checks [mV]:', [(lab, round(1000 * x, 1)) for _, x, lab in deltas])
P = run('p', deltas); NP = run('np', deltas) if __import__('os').path.exists(f'{D}/np/points.csv') else []
for s in P + NP:
    print(f"{s['sub']:2s} lap{s['lap']} D{s['duty']}: omega {s['w']:.0f}{'' if s['w_ok'] else ' (BAD LINE)'} V {s['V']:.4f} I_m {s['I_m']:.4f}"
          f"  K_e {s['ke']:.4e} (n={s['n']}, spread {s['spread']:.1f} %, coast {s['gap']:.0f} ms)  R_out {s['R']:.3f}")
if NP:
    print('\nduty  R_out no-prop (laps) | prop (laps) | ratio-1 [%]   (each step with its own K_e)')
    for d in (20, 30, 40, 50, 60):
        a = [s['R'] for s in NP if s['duty'] == d and np.isfinite(s['R'])]
        b = [s['R'] for s in P if s['duty'] == d and np.isfinite(s['R'])]
        if a and b:
            print(f"D{d}: {' '.join(f'{x:.3f}' for x in a)} | {' '.join(f'{x:.3f}' for x in b)} | {100 * (np.mean(a) / np.mean(b) - 1):+.1f}")
