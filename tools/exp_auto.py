#!/usr/bin/env python3
"""One fully automatic PWM step: battery reading, motor run, scope capture, scale photos, audio.
PWM 試験 1 段の全自動実行：電池電圧、モータ駆動、オシロ取り込み、秤の撮影、録音。

Needs sf_serial_bridge.py running (holds the StampFly port open).
Usage: python3 tools/exp_auto.py <tag> <motor 1-4> <duty %> [--sec 8] [--data DIR] [--scope-ip IP]
Scale value = mean(spin photos) - (before + after)/2, which cancels a linear drift of the scale.
The photos are read afterwards (by eye) and the readings are written to thrust_readings.json.
写真は後から目視で読み、thrust_readings.json に書く。
"""
import argparse, json, os, pathlib, re, subprocess, sys, threading, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import exp_rpm as E

# Bridge folder (same default as sf_serial_bridge.py) / 中継のフォルダ（sf_serial_bridge.py と同じ既定値）
BR = pathlib.Path(os.environ.get('SF_BRIDGE_DIR', str(pathlib.Path.home() / '.cache' / 'sf_serial_bridge')))
# The StampFly firmware INA3221 driver assumes a 0.1 ohm shunt, but the board's R12 is 0.01 ohm (schematic),
# so the reported current is 1/10 of the true value: true current = reported x I_SCALE.
# StampFly ファームの INA3221 ドライバはシャント 0.1 Ω を仮定するが、基板の R12 は 0.01 Ω（回路図）なので
# 表示電流は実際の 1/10。実電流 = 表示 × I_SCALE。環境変数 SF_CURRENT_SCALE で変更（既定 10.0）。
I_SCALE = float(os.environ.get('SF_CURRENT_SCALE', 10.0))
CAM = os.environ.get('SF_CAM_NAME', 'AtomS3R-CAM')               # USB camera on the scale LCD / 秤の表示を撮る USB カメラ
MIC_NAME = os.environ.get('SF_MIC_NAME', '')   # name of the microphone device (e.g. iPhone via Continuity); empty = no audio / マイク名、空なら録音しない


def mic_index():
    # look up the audio device index by name (order changes) / 番号は変わるので名前で探す
    import re
    out = subprocess.run(['ffmpeg', '-hide_banner', '-f', 'avfoundation', '-list_devices', 'true', '-i', ''],
                         capture_output=True, text=True).stderr
    aud = out.split('audio devices')[-1]
    for m in re.finditer(r'\[(\d+)\] (.*)', aud):
        if MIC_NAME in m.group(2):
            return m.group(1)
    return None


def send(cmd):
    with open(BR / 'cmd.fifo', 'w') as f:
        f.write(cmd + '\n')


def power():
    # read battery via 'sensor power' / 電池電圧を読む
    n0 = (BR / 'serial.log').stat().st_size
    send('sensor power'); time.sleep(0.8)
    txt = (BR / 'serial.log').read_bytes()[n0:].decode(errors='replace')
    for line in txt.splitlines():
        if line.startswith('power:'):
            return line.strip()
    return None


def parse_power(line):
    # 'power: 4.32 V  40 mA  173 mW' -> (V, A corrected) / 電圧と補正後の電流
    try:
        p = line.split()
        return float(p[1]), float(p[3]) * I_SCALE / 1000
    except Exception:
        return None


def poll_power(store, t_end):
    while time.time() < t_end:
        r = power()
        if r:
            store.append((round(time.time(), 2), r))
        time.sleep(0.4)


def photo(path, n=3):
    tile = {3: '3x1', 9: '3x3'}[n]
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'avfoundation', '-framerate', '15',
                    '-video_size', '640x480', '-i', f'{CAM}:none', '-ss', '0.8', '-frames:v', '1',
                    '-vf', f'fps=3.3,tile={tile}', str(path)], check=False)   # 3 frames 0.3 s apart / 0.3 秒おき 3 枚を横並び


def audio(path, sec):
    mic = mic_index() if MIC_NAME else None
    if mic is None:
        return
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'avfoundation', '-i', f':{mic}',
                    '-t', str(sec), '-ac', '1', '-ar', '48000', str(path)], check=False)


def blade_freq(wav):
    # dominant tone 100-1500 Hz in the recording / 録音の 100〜1500 Hz の最大の音
    import wave
    w = wave.open(str(wav)); x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float)
    fs = w.getframerate(); x -= x.mean()
    F = np.abs(np.fft.rfft(x * np.hanning(x.size))); f = np.fft.rfftfreq(x.size, 1 / fs)
    m = (f > 100) & (f < 1500)
    order = np.argsort(F[m])[::-1][:5]
    return [(round(float(f[m][i]), 1), round(float(F[m][i] / F[m][order[0]]), 2)) for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tag'); ap.add_argument('motor', type=int); ap.add_argument('duty', type=int)
    ap.add_argument('--sec', type=float, default=8.0)
    ap.add_argument('--audio', action='store_true', help='record the microphone (needs SF_MIC_NAME) / マイクで録音')
    E.add_common_args(ap)
    a = ap.parse_args()
    E.apply_common_args(a)
    E.check_tag(a.tag)
    if not (1 <= a.motor <= 4 and 0 <= a.duty <= 100):
        raise SystemExit('motor must be 1-4 and duty 0-100')   # firmware CLI range / ファームの範囲
    out = E.OUT; img = out / 'img'; img.mkdir(exist_ok=True)
    rec = {'motor': a.motor, 'duty_cmd': a.duty / 100, 'sec': a.sec}
    rec['power_before'] = power()
    photo(img / f'{a.tag}_0before.jpg')
    sc = E.Scope(); E.setup(sc, 'pwm')
    t0 = time.time(); send(f'motor test {a.motor} {a.duty} {a.sec:g}')
    time.sleep(2.5)
    pw = []
    tp = threading.Thread(target=poll_power, args=(pw, t0 + a.sec - 1.0)); tp.start()
    sc.w(':SING')                                   # 48 ms acquisition / 48 ms の取り込み
    th = threading.Thread(target=audio if a.audio else (lambda *x: None), args=(img / f'{a.tag}.wav', 2.5)); th.start()
    photo(img / f'{a.tag}_1spin.jpg', n=9)          # 9 frames over ~2.7 s while spinning / 回転中に約 2.7 秒で 9 枚
    rec['spin_photo_end_s'] = round(time.time() - t0, 2)
    th.join(); tp.join()
    vals = [parse_power(x[1]) for x in pw if parse_power(x[1])]
    rec['power_spin_samples'] = [x[1] for x in pw]
    if vals:
        rec['Vbat_spin_V'] = float(np.mean([v for v, _ in vals]))
        rec['Ibat_spin_A'] = float(np.mean([i for _, i in vals]))   # x10 corrected / 10 倍補正済み
    while time.time() - t0 < a.sec + 2.5:
        time.sleep(0.2)
    photo(img / f'{a.tag}_9after.jpg')
    rec['power_after'] = power()
    for k in range(50):
        if sc.q(':TRIG:STAT?') == 'STOP':
            break
        if k == 10:
            sc.w(':TFOR')            # no trigger (motor not running) / トリガなしなら強制
        time.sleep(0.2)
    t, v1, r1 = sc.grab_raw(1); _, v2, r2 = sc.grab_raw(2); sc.w(':RUN')
    np.savez(out / f'{a.tag}.npz', t=t, v1=v1, v2=v2, mode='pwm')
    r, _ = E.analyse({'t': t, 'v1': v1, 'v2': v2}, 'pwm')
    # measured duty from the drain (low = ON) / ドレインから実 Duty
    r['duty_meas'] = float(np.mean(v1 < 1.0))
    r['clip'] = [int(x.min() == 0 or x.max() == 255) for x in (r1, r2)]
    try:
        r['audio_peaks_Hz_rel'] = blade_freq(img / f'{a.tag}.wav')
    except Exception as e:
        r['audio_peaks_Hz_rel'] = str(e)
    rec.update(r); rec['time'] = time.strftime('%H:%M:%S')
    resf = out / 'results.json'
    res = json.loads(resf.read_text()) if resf.exists() else {}
    res[a.tag] = rec; resf.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rec.items() if k != 'peaks_Hz_rel'}, ensure_ascii=False))
    print('ripple peaks [Hz, rel]:', [(round(p, 1), round(q, 2)) for p, q in r['peaks_Hz_rel']])


if __name__ == '__main__':
    main()
