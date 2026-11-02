#!/usr/bin/env python3
"""Capture and analyse Ke / rotor speed / thrust experiment.
K_e・回転数・推力の同時測定の取り込みと解析。

Usage:
  python3 tools/exp_rpm.py dc  <tag> [--thrust G]   # DC drive: CH1 = shunt 0.1 ohm (low side), CH2 = motor +
  python3 tools/exp_rpm.py pwm <tag> [--thrust G]   # PWM (real board): CH1 = drain, CH2 = V_DD (cable split)
  python3 tools/exp_rpm.py summary                   # table of all captures
Data: <data dir>/<tag>.npz and results.json (data dir: --data DIR or env AUTOTEST_DATA, default ./data).
Scope: --scope-ip IP or env SCOPE_IP (no default).
出力先は --data / 環境変数 AUTOTEST_DATA（既定 ./data）、オシロの IP は --scope-ip / SCOPE_IP（既定なし）。
Ripple frequency f_comm from FFT (DC: of the shunt voltage; PWM: of per-PWM-period means of V_C = V_DD - v_D).
Rotor speed: omega = 2*pi*f_comm/N (N set with --N once known; f_rev line reported to determine N).
"""
import argparse, json, os, pathlib, re, socket, time
import numpy as np

IP = os.environ.get('SCOPE_IP', '')   # oscilloscope IP, no default / オシロの IP（既定なし）
OUT = pathlib.Path(os.environ.get('AUTOTEST_DATA', 'data'))   # output folder / 出力フォルダ


def set_out(path=None):
    # select and create the output folder / 出力フォルダを決めて作る
    global OUT
    if path:
        OUT = pathlib.Path(path)
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def set_scope_ip(ip=None):
    global IP
    if ip:
        IP = ip


def add_common_args(ap):
    ap.add_argument('--data', help='output folder (env AUTOTEST_DATA, default ./data) / 出力フォルダ')
    ap.add_argument('--scope-ip', help='oscilloscope IP (env SCOPE_IP) / オシロの IP')


def apply_common_args(a, need_scope=True):
    set_out(a.data); set_scope_ip(a.scope_ip)
    if need_scope and not IP:
        raise SystemExit('oscilloscope IP is not set: use --scope-ip IP or env SCOPE_IP / オシロの IP が未設定です')

RSH = 0.1                 # shunt [ohm] / シャント抵抗
F_PWM = float(os.environ.get('PWM_FREQ_HZ', 150e3))   # PWM frequency [Hz] / PWM 周波数
# CH2 probe correction: delay [ns], gain, offset [V]. Default = no correction.
# The values come from a same-node test of one's own probe pair (connect CH1 and CH2 to the same node
# and compare); set them with env CH2_DELAY_NS, CH2_GAIN, CH2_OFFSET_V.
# CH2 の補正（遅延 ns、ゲイン、オフセット V）。既定は補正なし。自分のプローブ 2 本を同じ節点につないで
# 比べる試験（same-node test）から求めた値を環境変数で与える。
SH = float(os.environ.get('CH2_DELAY_NS', 0.0))
G = float(os.environ.get('CH2_GAIN', 1.0))
O = float(os.environ.get('CH2_OFFSET_V', 0.0))


def check_tag(tag):
    # tags become file names: letters, digits, _ . - only / タグはファイル名になるので英数字と _ . - のみ
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', tag or '') or tag.startswith('.'):
        raise SystemExit(f'invalid tag: {tag!r}')


class Scope:
    def __init__(self):
        self.s = socket.create_connection((IP, 5555), timeout=30)

    def w(self, c):
        self.s.sendall((c + '\n').encode())

    def q(self, c):
        self.w(c); d = b''
        while not d.endswith(b'\n'):
            d += self.s.recv(4096)
        return d.decode().strip()

    def block(self):
        h = b''
        while len(h) < 2:
            h += self.s.recv(2 - len(h))
        nd = int(chr(h[1])); ln = b''
        while len(ln) < nd:
            ln += self.s.recv(nd - len(ln))
        n = int(ln); d = b''
        while len(d) < n + 1:
            d += self.s.recv(1 << 20)
        return d[:n]

    def grab_raw(self, ch):
        # read the full acquisition memory in STOP state / 停止状態でメモリ全体を読む
        self.w(f':WAV:SOUR CHAN{ch}'); self.w(':WAV:MODE RAW'); self.w(':WAV:FORM BYTE')
        pre = [float(x) for x in self.q(':WAV:PRE?').split(',')]
        npts = int(pre[2]); parts = []
        for a in range(1, npts + 1, 250000):
            self.w(f':WAV:STAR {a}'); self.w(f':WAV:STOP {min(a + 249999, npts)}'); self.w(':WAV:DATA?')
            parts.append(self.block())
        y = np.frombuffer(b''.join(parts), dtype=np.uint8).astype(float)
        xinc, xor, xref, yinc, yor, yref = pre[4:10]
        return np.arange(len(y)) * xinc + xor, (y - yor - yref) * yinc, y


def channels_on(sc):
    # CH1 and CH2 on, CH3 and CH4 off: the scripts read CH1/CH2, and two channels allow the 60k/600k/6M depths
    # (after a self-calibration on 2026-11-02 CH2 was left off, so CH2 would have been recorded as garbage)
    # CH1・CH2 を ON、CH3・CH4 を OFF。スクリプトは CH1/CH2 を読み、2 チャネルなら 60k/600k/6M が使える
    # （2026-11-02 の自己較正の後、CH2 が OFF のままで、そのままでは CH2 が無意味な値で記録されていた）
    for c, on in ((1, 'ON'), (2, 'ON'), (3, 'OFF'), (4, 'OFF')):
        sc.w(f':CHAN{c}:DISP {on}')
    time.sleep(0.3)


# Vertical settings of the PWM captures. Default: CH1 1.25 V/div -3.25 V, CH2 0.5 V/div -3.5 V (2026-10-15 and the
# morning of 2026-11-02). SCOPE_MATCHED=1: both channels 1 V/div -2.25 V, so the two front ends share one gain path
# and V_DD - v_D does not carry a channel-to-channel gain difference (calibration of 2026-11-02 afternoon).
# PWM 取り込みの縦軸設定。既定は CH1 1.25 V/div・-3.25 V、CH2 0.5 V/div・-3.5 V（10-15 と 11-02 午前）。
# SCOPE_MATCHED=1 で両チャネル 1 V/div・-2.25 V。同じ増幅経路になり、V_DD - v_D にチャネル間の倍率差が入らない
MATCHED = os.environ.get('SCOPE_MATCHED', '0') == '1'


def vertical_pwm(sc):
    # CH1 drain (0..4.4 V) and CH2 V_DD (3.5..4.4 V) / CH1 ドレイン、CH2 V_DD
    if MATCHED:
        sc.w(':CHAN1:SCAL 1'); sc.w(':CHAN1:OFFS -2.25'); sc.w(':CHAN2:SCAL 1'); sc.w(':CHAN2:OFFS -2.25')
    else:
        sc.w(':CHAN1:SCAL 1.25'); sc.w(':CHAN1:OFFS -3.25'); sc.w(':CHAN2:SCAL 0.5'); sc.w(':CHAN2:OFFS -3.5')


def setup(sc, mode):
    sc.w(':RUN'); channels_on(sc); sc.w(':ACQ:TYPE NORM'); sc.w(':TRIG:SWE AUTO')
    if mode == 'dc':
        # CH1 shunt (up to ~0.3 V), CH2 motor + (up to ~4.3 V) / CH1 シャント、CH2 モータ + 端子
        sc.w(':CHAN1:SCAL 0.05'); sc.w(':CHAN1:OFFS -0.15'); sc.w(':CHAN2:SCAL 0.5'); sc.w(':CHAN2:OFFS -2.5')
        sc.w(':TIM:MAIN:SCAL 2e-3'); sc.w(':TIM:MAIN:OFFS 0'); time.sleep(0.3); sc.w(':ACQ:MDEP 600000')
    else:
        # CH1 drain, CH2 V_DD / CH1 ドレイン、CH2 V_DD
        vertical_pwm(sc)
        sc.w(':TIM:MAIN:SCAL 2e-3'); sc.w(':TIM:MAIN:OFFS 0'); time.sleep(0.3); sc.w(':ACQ:MDEP 6000000')
    time.sleep(1.5)


def spectrum(x, fs, fmin=150., fmax=12e3):
    x = x - x.mean(); win = np.hanning(x.size)
    F = np.abs(np.fft.rfft(x * win)); f = np.fft.rfftfreq(x.size, 1 / fs)
    m = (f > fmin) & (f < fmax)
    return f[m], F[m]


def analyse(d, mode, N=None):
    t, v1, v2 = d['t'], d['v1'], d['v2']; fs = 1 / (t[1] - t[0])
    r = {'fs': fs, 'T_rec_ms': (t[-1] - t[0]) * 1e3}
    if mode == 'dc':
        I = v1 / RSH; vm = v2 - v1                       # motor current and motor voltage / 電流とモータ電圧
        r.update(I_A=float(I.mean()), Vm_V=float(vm.mean()), I_pp=float(np.ptp(I)))
        sig, fsig = I, fs
    else:
        v2c = (np.interp(t, t + SH * 1e-9, v2) - O) / G; vc = v2c - v1
        n = int(round(fs / F_PWM)); n_eff = fs / F_PWM
        # per-PWM-period means (window = one PWM period; resample if fs/F_PWM not integer)
        # PWM 1 周期ごとの平均（窓を周期ちょうどに。整数でなければ補間）
        tt = np.arange(t[0], t[-1] - 1 / F_PWM, 1 / F_PWM)
        cum = np.concatenate([[0], np.cumsum(vc)]) / fs
        idx0 = (tt - t[0]) * fs; idx1 = idx0 + n_eff
        c0 = np.interp(idx0, np.arange(cum.size), cum); c1 = np.interp(idx1, np.arange(cum.size), cum)
        avg = (c1 - c0) * F_PWM
        r.update(VC_mean_V=float(vc.mean()), VDD_mean_V=float(v2c.mean()), n_per_period=n_eff)
        sig, fsig = avg, F_PWM
    f, F = spectrum(sig, fsig)
    order = np.argsort(F)[::-1]
    peaks = []
    for i in order:
        if all(abs(f[i] - p) > 3 * (f[1] - f[0]) for p, _ in peaks):
            peaks.append((float(f[i]), float(F[i] / F[order[0]])))
        if len(peaks) >= 6:
            break
    r['peaks_Hz_rel'] = peaks
    r['f_comm_Hz'] = peaks[0][0]
    if N:
        r['N'] = N; r['omega_rad_s'] = 2 * np.pi * peaks[0][0] / N; r['rpm'] = peaks[0][0] / N * 60
    return r, (f, F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['dc', 'pwm', 'summary'])
    ap.add_argument('tag', nargs='?')
    ap.add_argument('--thrust', type=float, help='scale reading [g] / 秤の読み')
    ap.add_argument('--N', type=int, help='ripples per revolution / 1 回転あたりの脈動の数')
    ap.add_argument('--nosetup', action='store_true')
    add_common_args(ap)
    a = ap.parse_args()
    apply_common_args(a, need_scope=a.mode != 'summary')
    resf = OUT / 'results.json'
    res = json.loads(resf.read_text()) if resf.exists() else {}
    if a.mode == 'summary':
        for k, v in res.items():
            print(k, {kk: v[kk] for kk in v if kk not in ('peaks_Hz_rel',)}, 'peaks', [(round(p), round(r, 2)) for p, r in v.get('peaks_Hz_rel', [])[:4]])
        return
    check_tag(a.tag)
    sc = Scope()
    if not a.nosetup:
        setup(sc, a.mode)
    sc.w(':SING')
    for _ in range(100):
        time.sleep(0.2)
        if sc.q(':TRIG:STAT?') == 'STOP':
            break
        if _ == 25:
            sc.w(':TFOR')            # no trigger edge (e.g. motor off): force one / トリガがなければ強制
    else:
        sc.w(':STOP'); time.sleep(0.5)
    t, v1, r1 = sc.grab_raw(1); _, v2, r2 = sc.grab_raw(2); sc.w(':RUN')
    clip = [int(r.min() == 0 or r.max() == 255) for r in (r1, r2)]
    np.savez(OUT / f'{a.tag}.npz', t=t, v1=v1, v2=v2, mode=a.mode, thrust=np.nan if a.thrust is None else a.thrust)
    r, (f, F) = analyse({'t': t, 'v1': v1, 'v2': v2}, a.mode, a.N)
    r.update(mode=a.mode, thrust_g=a.thrust, clip=clip, time=time.strftime('%H:%M:%S'))
    res[a.tag] = r; resf.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: r[k] for k in r if k != 'peaks_Hz_rel'}, ensure_ascii=False))
    print('peaks [Hz, rel]:', [(round(p, 1), round(q, 2)) for p, q in r['peaks_Hz_rel']])


if __name__ == '__main__':
    main()
