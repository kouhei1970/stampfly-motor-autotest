#!/usr/bin/env python3
"""Probe and sensor calibration at rest: CH1, CH2 and the CLI voltage/current at the run settings.
静止時のプローブ・センサの校正: 本番の設定のままの CH1・CH2 と CLI の電圧・電流。

Put the probes where the label says (e.g. both on V_DD for a same-node test, or CH1 on the drain), keep the
motor stopped, then run:  python3 tools/scope_cal.py <label> [--n 3] [--data DIR] [--scope-ip IP]
Appends one row per capture to <data>/scope_cal.csv. Settings = exp_rpm setup 'pwm' with --mdep (exp_long default),
plus one capture with CH1 at the CH2 settings and one with CH1 offset 0 (to separate the offset-setting error).
プローブをラベルのとおりに置き（同じ点の試験なら両方 V_DD、ドレインなら CH1 をドレイン）、モータを止めて実行する。
"""
import argparse, csv, os, re, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import exp_rpm as E
import exp_auto as A


def grab(sc):
    # one forced single capture, channel means / 強制トリガで 1 回取り込み、各チャネルの平均
    sc.w(':SING')
    for k in range(60):
        if sc.q(':TRIG:STAT?') == 'STOP':
            break
        if k == 5:
            sc.w(':TFOR')
        time.sleep(0.05)
    _, v1, _ = sc.grab_raw(1); _, v2, _ = sc.grab_raw(2); sc.w(':RUN')
    return float(v1.mean()), float(v2.mean())


def cli():
    # mean CLI voltage and current over three readings / CLI の電圧・電流の 3 回平均
    vals = [A.parse_power(A.power()) for _ in range(3)]
    vals = [v for v in vals if v]
    return (float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals]))) if vals else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('label'); ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--mdep', type=int, default=60000)
    E.add_common_args(ap); a = ap.parse_args(); E.apply_common_args(a)
    sc = E.Scope(); E.setup(sc, 'pwm'); sc.w(f':ACQ:MDEP {a.mdep}'); time.sleep(1.5)
    out = E.OUT / 'scope_cal.csv'; new = not out.exists()
    with open(out, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['wall_time_s', 'label', 'ch1_setting', 'CH1_V', 'CH2_V', 'CH1_minus_CH2_mV', 'CLI_V', 'CLI_minus_CH2_mV', 'CLI_I_A'])
        rows = []
        for setting, cmds in (('run (matched)' if E.MATCHED else 'run (1.25 V/div, -3.25 V)', []), ('as CH2 (0.5 V/div, -3.5 V)', [':CHAN1:SCAL 0.5', ':CHAN1:OFFS -3.5']),
                              ('1.25 V/div, offset 0', [':CHAN1:SCAL 1.25', ':CHAN1:OFFS 0'])):
            for c in cmds:
                sc.w(c)
            if cmds:
                time.sleep(1.0)
            for _ in range(a.n if not cmds else 1):
                c1, c2 = grab(sc); cv, ci = cli()
                row = [f'{time.time():.2f}', a.label, setting, f'{c1:.4f}', f'{c2:.4f}', f'{1000 * (c1 - c2):+.1f}',
                       f'{cv:.3f}', f'{1000 * (cv - c2):+.1f}', f'{ci:.4f}']
                w.writerow(row); f.flush(); rows.append(row)
                print(' '.join(row[1:]))
        E.vertical_pwm(sc)                                      # back to the run settings / 本番の設定に戻す
    print('appended to', out)


if __name__ == '__main__':
    main()
