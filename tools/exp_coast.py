#!/usr/bin/env python3
"""Spin-down (coast) test for the rotor inertia J.
惰性回転（スピンダウン）試験で慣性モーメント J を求める。

After `motor stop` the MOSFET stays OFF; with C_C only across the motor there is no current path,
so the motor current is 0 and V_DD - v_D = K_e * omega (back-EMF).  Then
  J d(omega)/dt = -(C_Q omega^2 + K_e I0(omega))
MOSFET を OFF にすると C_C しかつながらず電流は 0、端子電圧 = 逆起電力。上の運動方程式から J を出す。

Usage: python3 tools/exp_coast.py <tag> <duty %> <run s> <timebase s/div> [--motor N] [--data DIR] [--scope-ip IP]
"""
import argparse, json, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import exp_rpm as E
import exp_auto as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tag'); ap.add_argument('duty', type=int); ap.add_argument('run', type=float)
    ap.add_argument('tb', type=float); ap.add_argument('--motor', type=int, default=1)
    E.add_common_args(ap)
    a = ap.parse_args()
    E.apply_common_args(a)
    E.check_tag(a.tag)
    if not (1 <= a.motor <= 4 and 0 <= a.duty <= 100):
        raise SystemExit('motor must be 1-4 and duty 0-100')   # firmware CLI range / ファームの範囲
    sc = E.Scope(); sc.w(':RUN')
    sc.w(':CHAN1:SCAL 1.25'); sc.w(':CHAN1:OFFS -3.25'); sc.w(':CHAN2:SCAL 0.5'); sc.w(':CHAN2:OFFS -3.5')
    sc.w(':TRIG:MODE EDGE'); sc.w(':TRIG:EDG:SOUR CHAN1'); sc.w(':TRIG:EDG:SLOP NEG'); sc.w(':TRIG:EDG:LEV 2.0')
    sc.w(':TRIG:SWE NORM')
    sc.w(f':TIM:MAIN:SCAL {a.tb}'); sc.w(f':TIM:MAIN:OFFS {5.5 * a.tb}')   # trigger near left edge / トリガを左端付近に
    time.sleep(0.5); sc.w(':ACQ:MDEP 6000000'); time.sleep(1.0)
    rec = {'duty_cmd': a.duty / 100, 'run_s': a.run, 'tb': a.tb, 'power_before': A.power()}
    A.send(f'motor test {a.motor} {a.duty} {a.run + 5:g}')
    time.sleep(a.run)
    rec['power_spin'] = A.power()
    sc.w(':SING'); time.sleep(0.4)
    A.send('motor stop'); t_stop = time.time()
    for _ in range(200):
        time.sleep(0.2)
        if sc.q(':TRIG:STAT?') == 'STOP':
            break
    t, v1, r1 = sc.grab_raw(1); _, v2, r2 = sc.grab_raw(2)
    sc.w(':TRIG:SWE AUTO'); sc.w(':RUN')
    np.savez(E.OUT / f'{a.tag}.npz', t=t, v1=v1, v2=v2, mode='coast')
    rec['clip'] = [int(x.min() == 0 or x.max() == 255) for x in (r1, r2)]
    rec['time'] = time.strftime('%H:%M:%S')
    resf = E.OUT / 'results.json'
    res = json.loads(resf.read_text()); res[a.tag] = rec; resf.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(rec, ensure_ascii=False))


if __name__ == '__main__':
    main()
