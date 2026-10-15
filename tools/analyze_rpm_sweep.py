#!/usr/bin/env python3
"""Summarise a PWM sweep: rotor speed (ripple, N = 10), thrust, battery current.
PWM 掃引のまとめ：回転数（整流の脈動、N = 10）、推力、電池電流。

Thrust readings (read from the scale photos) are in <data dir>/thrust_readings.json:
  {tag: {"before": [..], "spin": [..], "after": [..]}}  [g]
Output: summary.json, summary.png (same folder).
Usage: python3 tools/analyze_rpm_sweep.py [--data DIR]  (env AUTOTEST_DATA, default ./data)
"""
import argparse, json, os, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import exp_rpm as E

N = 10                      # ripples per revolution (audio 1/rev and blade-pass lines) / 1 回転あたりの脈動の数
E.F_PWM = float(os.environ.get('PWM_FREQ_HZ', 150096.67))   # measured PWM frequency (80 MHz / 533) / 実測の PWM 周波数


D = None   # data folder, set in main() / データフォルダ（main で設定）


def fine_peak(sig, fs, f0, half=150.0, pad=16):
    # zero-padded FFT, parabolic interpolation near f0 / ゼロ詰め FFT で f0 付近の線を細かく読む
    x = (sig - sig.mean()) * np.hanning(sig.size)
    n = sig.size * pad
    F = np.abs(np.fft.rfft(x, n)); f = np.fft.rfftfreq(n, 1 / fs)
    m = np.flatnonzero((f > f0 - half) & (f < f0 + half))
    k = m[np.argmax(F[m])]
    a, b, c = np.log(F[k - 1:k + 2])
    return float(f[k] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0]))


def per_period_mean(t, v1, v2):
    fs = 1 / (t[1] - t[0])
    v2c = (np.interp(t, t + E.SH * 1e-9, v2) - E.O) / E.G
    vc = v2c - v1
    tt = np.arange(t[0], t[-1] - 1 / E.F_PWM, 1 / E.F_PWM)
    cum = np.concatenate([[0], np.cumsum(vc)]) / fs
    i0 = (tt - t[0]) * fs; i1 = i0 + fs / E.F_PWM
    idx = np.arange(cum.size)
    return (np.interp(i1, idx, cum) - np.interp(i0, idx, cum)) * E.F_PWM, float(v2c.mean()), float(vc.mean())


def main():
    global D
    ap = argparse.ArgumentParser(); ap.add_argument('--data', help='data folder / データフォルダ')
    D = E.set_out(ap.parse_args().data)
    res = json.loads((D / 'results.json').read_text())
    thr = json.loads((D / 'thrust_readings.json').read_text()) if (D / 'thrust_readings.json').exists() else {}
    rows = []
    for tag in sorted(k for k in res if k.startswith('pwm_')):
        r = res[tag]; d = np.load(D / f'{tag}.npz')
        avg, vdd, vc = per_period_mean(d['t'], d['v1'], d['v2'])
        f_coarse = r['peaks_Hz_rel'][0][0]
        f_comm = fine_peak(avg, E.F_PWM, f_coarse)
        row = dict(tag=tag, duty=r['duty_cmd'], duty_meas=r['duty_meas'], VDD=vdd, Vm_avg=vc,
                   Ibat=r.get('Ibat_spin_A'), f_comm=f_comm, omega=2 * np.pi * f_comm / N,
                   audio=r.get('audio_peaks_Hz_rel'))
        if tag in thr:
            t_ = thr[tag]
            base = (np.mean(t_['before']) + np.mean(t_['after'])) / 2
            row.update(thrust_g=float(np.mean(t_['spin']) - base), thrust_sd=float(np.std(t_['spin'])),
                       drift_g=float(np.mean(t_['after']) - np.mean(t_['before'])))
            F = row['thrust_g'] * 9.80665e-3
            row['C_T'] = F / row['omega'] ** 2
        rows.append(row)
    (D / 'summary.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    for w in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in w.items() if k != 'audio'})
    plot(rows)


def plot(rows):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Hiragino Sans', 'DejaVu Sans']
    main_ = [r for r in rows if not r['tag'].endswith('_rep')]; rep = [r for r in rows if r['tag'].endswith('_rep')]
    D_ = np.array([r['duty_meas'] for r in main_])
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    a = ax[0, 0]
    a.plot(D_, [r['omega'] for r in main_], 'o-', label='整流の脈動（N=10）')
    a.plot([r['duty_meas'] for r in rep], [r['omega'] for r in rep], 's', ms=9, mfc='none', label='D0.5 再測定')
    a.set_xlabel('Duty（ドレイン波形から）'); a.set_ylabel('ω [rad/s]'); a.set_title('回転数'); a.legend(fontsize=8)
    a = ax[0, 1]
    if all('thrust_g' in r for r in main_):
        a.errorbar(D_, [r['thrust_g'] for r in main_], [r['thrust_sd'] for r in main_], fmt='o-', capsize=3, label='今回の測定')
        a.plot([r['duty_meas'] for r in rep], [r['thrust_g'] for r in rep], 's', ms=9, mfc='none', label='D0.5 再測定')
        a.legend(fontsize=8)
    a.set_xlabel('Duty'); a.set_ylabel('推力 [g]'); a.set_title('推力（回転中 − 回転前後の平均）')
    a = ax[1, 0]
    a.plot(D_, [r['Ibat'] for r in main_], 'o-'); a.set_xlabel('Duty'); a.set_ylabel('電池電流 [A]')
    a.set_title('電池電流（INA3221、表示 × 10 に補正）')
    a = ax[1, 1]
    if all('C_T' in r for r in main_):
        om = np.array([r['omega'] for r in main_]); Fn = np.array([r['thrust_g'] for r in main_]) * 9.80665e-3
        a.plot(om, Fn * 1e3, 'o', label='測定')
        w = np.linspace(0, om.max() * 1.05, 100)
        ct = float(np.sum(Fn * om ** 2) / np.sum(om ** 4))
        a.plot(w, ct * w ** 2 * 1e3, '-', label=f'当てはめ C_T = {ct:.2e}')
        a.set_xlabel('ω [rad/s]'); a.set_ylabel('推力 [mN]'); a.set_title('推力と回転数'); a.legend(fontsize=8)
    for a in ax.flat:
        a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(D / 'summary.png', dpi=110)


if __name__ == '__main__':
    main()
