#!/usr/bin/env python3
"""V_DD at M1 while M1 is on and while it is off, per step, and the one-motor / four-motor comparison.
M1 がオンの間とオフの間の V_DD（段ごと）と、1 モータ・4 モータ同時の比較。

With four motors on the same PWM timer, all four switch on together, so the supply dips further during the
on-time than with one motor; the mean V_DD then overstates the voltage the motor is driven with.
4 つのモータは同じタイマで同時にオンになるので、オンの間の電圧の落ち込みが 1 モータより大きい。
そのため平均の V_DD は、モータを駆動している電圧より高めになる。

Reads each step's steady capture (<run>/<npz> from points.csv; CH1 = M1 drain, CH2 = V_DD) and adds the
columns VDD_on_V and VDD_off_V to data/2026-11-02_steps.csv. Then compares the four-motor M1 speed with the
one-motor map omega(D, V) built on the mean V_DD and on the on-time V_DD, only where the one-motor data cover
the voltage. The raw captures are not in the repository; the columns it writes are.
各段の定常の取り込みを読み、data/2026-11-02_steps.csv に VDD_on_V・VDD_off_V の列を足す。続けて、4 モータの M1 の
回転数を、平均の V_DD とオンの間の V_DD で作った 1 モータの表と比べる（1 モータのデータがある電圧の範囲だけ）。

Usage: python3 tools/vdd_on_time.py [--data DIR]   (default ./data; the run folders must be present)
"""
import argparse, pathlib, sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import exp_rpm as E

ON_V = 0.2                  # drain below this = M1 on / ドレインがこれより低い = オン [V]
RUN_DIRS = {'bat2': '2026-11-02_bat2', 'bat3': '2026-11-02_bat3', 'R15': '2026-11-02_R15'}


def on_off_vdd(npz):
    # mean V_DD over the on-time and the off-time, one sample trimmed at each switching edge
    # オンの間とオフの間の V_DD の平均。切り替わりの前後 1 点ずつは除く
    d = np.load(npz); t, v1, v2 = d['t'], d['v1'], d['v2']
    v2c = (np.interp(t, t + E.SH * 1e-9, v2) - E.O) / E.G      # CH2 correction as in analyze_rpm_sweep / CH2 の補正
    on = v1 < ON_V
    core = on & np.roll(on, 1) & np.roll(on, -1)
    off = ~(on | np.roll(on, 1) | np.roll(on, -1))
    return float(v2c[core].mean()), float(v2c[off].mean())


def fitmap(D, V, y):
    # omega = b0 + b1 D + b2 V + b3 D V + b4 D^2 / 回転数の表の形
    X = np.c_[np.ones_like(D), D, V, D * V, D ** 2]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return lambda d, v: np.c_[np.ones_like(d), d, v, d * v, d ** 2] @ b


def compare(st):
    # four-motor M1 against the one-motor map, mean V_DD and on-time V_DD / 4 モータの M1 を 1 モータの表と比べる
    one, four = st[st.battery == 'bat2'], st[st.battery == 'bat3']
    for label, col in (('mean V_DD', 'VDD_V'), ('on-time V_DD', 'VDD_on_V')):
        m = fitmap(one.duty.values, one[col].values, one.omega_rad_s.values)
        fit = one.omega_rad_s.values / m(one.duty.values, one[col].values) - 1
        r = four.omega_rad_s.values / m(four.duty.values, four[col].values) - 1
        inside = (four[col] >= one[col].min()).values & (four[col] <= one[col].max()).values
        print(f'{label}: one-motor fit rms {100 * np.sqrt(np.mean(fit ** 2)):.2f} %; four-motor steps inside '
              f'{inside.sum()}/{len(four)}: mean {100 * r[inside].mean():+.2f} %, rms {100 * np.sqrt(np.mean(r[inside] ** 2)):.2f} %')
    for name, g in (('one motor', one), ('four motors', four)):
        print(f'{name}: mean V_DD - on-time V_DD median {1000 * (g.VDD_V - g.VDD_on_V).median():.0f} mV, '
              f'off - on {1000 * (g.VDD_off_V - g.VDD_on_V).min():.0f}-{1000 * (g.VDD_off_V - g.VDD_on_V).max():.0f} mV')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default='data')
    a = ap.parse_args(); data = pathlib.Path(a.data)
    path = data / '2026-11-02_steps.csv'
    st = pd.read_csv(path)
    on_v, off_v = [], []
    for r in st.itertuples():
        run = data / RUN_DIRS[r.battery] / r.run
        p = pd.read_csv(run / 'points.csv')
        npz = p[(p.lap == r.lap) & (p.step == r.step)].npz
        if len(npz) != 1:
            sys.exit(f'no unique capture for {r.battery} {r.run} lap {r.lap} step {r.step}')
        v_on, v_off = on_off_vdd(run / npz.iloc[0])
        on_v.append(v_on); off_v.append(v_off)
    st['VDD_on_V'], st['VDD_off_V'] = on_v, off_v
    st.to_csv(path, index=False, float_format='%.6g')
    print(f'wrote VDD_on_V, VDD_off_V for {len(st)} steps to {path}')
    compare(st)


if __name__ == '__main__':
    main()
