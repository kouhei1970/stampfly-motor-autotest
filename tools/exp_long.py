#!/usr/bin/env python3
"""Long-discharge acquisition: duty laps (cycle) and duty steps (step), with continuous camera and audio.
長時間放電の取得：Duty 周回（cycle）と Duty ステップ（step）。カメラと録音は運転の間ずっと続ける。

Needs sf_serial_bridge.py running (holds the StampFly port open), unless --dry-run.
Usage:
  python3 tools/exp_long.py cycle <tag> --duties 50,60,...,80 --step-sec 5 --laps 1 [--total-sec 240]
        [--zero-stop-sec 4] [--wait-sec 1.5] [--vmin 3.5] [--motor 1] [--mdep 600000] [--audio] [--section A|B|C]
  python3 tools/exp_long.py step <tag> --steps 0:60,60:68,68:60,50:80,80:50,60:0 [--pre-sec 2.5] [--tb 0.1] [--mdep 12000000] [--keepalive-sec 10]
  common: [--fps 3] [--data DIR] [--scope-ip IP] [--dry-run [--dry-run-camera]]
Plan: docs/procedure_long_discharge.md (sections 6-8).
手順書：docs/procedure_long_discharge.md の 6〜8 章。
"""
import argparse, csv, json, os, pathlib, queue, signal, subprocess, sys, threading, time, atexit
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import exp_rpm as E
import exp_auto as A
import analyze_rpm_sweep as R      # fine_peak, per_period_mean, N, measured PWM frequency / 脈動の細密読み取りなど

FAKE_POWER = 'power: 4.10 V  700 mA  2870 mW'   # dry-run stand-in for 'sensor power' / dry-run 用の偽の電力行
FAKE_READOUT_S = 0.5                            # dry-run stand-in for the LAN readout time / dry-run の読み出し時間


def now():
    # wall clock rounded to 0.01 s / 壁時計（0.01 秒単位）
    return round(time.time(), 2)


# ---------------------------------------------------------------- command sink and event log
class Ctl:
    """Sends firmware CLI commands (or only logs them in dry-run) and records every one in events.csv.
    ファームの CLI 指令を送り（dry-run なら記録のみ）、すべて events.csv に残す。"""

    def __init__(self, dry):
        self.dry = dry
        self.lock = threading.RLock()        # re-entrant: the SIGINT handler may run while the main thread sends
        self.f = open(E.OUT / 'events.csv', 'w', newline='')
        self.w = csv.writer(self.f); self.w.writerow(['wall_time_s', 'kind', 'detail', 'note'])
        self.f.flush()

    def event(self, kind, detail='', note=''):
        # append one row to events.csv / events.csv に 1 行追加
        with self.lock:
            self.w.writerow([f'{time.time():.2f}', kind, detail, note]); self.f.flush()

    def send(self, cmd, note=''):
        # send a command; returns the send time / 指令を送り、送信時刻を返す
        with self.lock:
            if self.dry:
                print(f'[dry-run] {cmd}')
            else:
                A.send(cmd)
            t = time.time()
            self.event('cmd', cmd, (note + ' ' if note else '') + ('dry-run' if self.dry else ''))
            return t

    def power(self):
        # 'sensor power' line (fake in dry-run) / sensor power の行（dry-run では偽）
        if self.dry:
            return FAKE_POWER
        return A.power()


# ---------------------------------------------------------------- fake scope for --dry-run
class FakeScope:
    """Returns synthetic arrays shaped like Scope.grab_raw: drain square wave of the commanded duty with a small
    ripple at f_comm = 90 Hz x duty%, and V_DD about 4.0 V.  Sleeps FAKE_READOUT_S per readout.
    Scope.grab_raw と同じ形の合成波形を返す。ドレインは指令 Duty の矩形波に小さな脈動、V_DD は約 4.0 V。"""

    def __init__(self, n=600000, fs=12.5e6):
        self.n, self.fs = n, fs
        self.duty = 0.6            # commanded duty, set by the caller / 指令 Duty（呼び出し側が更新）
        self.step = None           # (pre, post) duty fractions for step mode / step 用の (前, 後)

    def w(self, c):
        pass

    def q(self, c):
        return 'STOP'

    def grab_raw(self, ch):
        time.sleep(FAKE_READOUT_S / 2)
        t = np.arange(self.n) / self.fs
        if self.step:
            duty = np.where(t < 0.1 * t[-1], self.step[0], self.step[1])
        else:
            duty = np.full(t.size, self.duty)
        f_comm = 90.0 * duty * 100
        ripple = 1 + 0.05 * np.sin(2 * np.pi * np.cumsum(f_comm) / self.fs)
        phase = (t * R.E.F_PWM) % 1.0
        v1 = np.where(phase < duty * ripple, 0.1, 4.0)
        v2 = np.full(t.size, 4.0) + 0.01 * np.random.randn(t.size)
        return t, (v1 if ch == 1 else v2), np.zeros(t.size)


# ---------------------------------------------------------------- continuous camera and microphone
class Camera:
    """Background ffmpeg: continuous scale photos, ~fps frames/s, saved as img/<tag>_<index>.jpg with arrival time.
    Arrival time = capture time + an unknown constant pipeline delay, to be measured from a duty step later.
    背景の ffmpeg で秤の写真を連続で撮り、到着時刻を frames.csv に残す。到着時刻 = 撮影時刻 + 未知の一定の
    パイプライン遅れ（後で Duty ステップから測る）。"""

    def __init__(self, tag, fps, ctl):
        self.tag, self.fps, self.ctl = tag, fps, ctl
        self.img = E.OUT / 'img'; self.img.mkdir(exist_ok=True)
        self.p = None; self.th = None; self.n = 0

    def start(self):
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'avfoundation', '-framerate', '15',
               '-video_size', '640x480', '-i', f'{A.CAM}:none', '-vf', f'fps={self.fps:g}',
               '-f', 'image2pipe', '-vcodec', 'mjpeg', '-q:v', '3', '-']
        # own session so a terminal Ctrl-C does not kill it abruptly; we stop it ourselves
        # 独自セッション（端末の Ctrl-C で突然死なせず、こちらで止める）
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True)
        atexit.register(self.kill)
        self.ctl.event('camera_start', A.CAM)
        self.th = threading.Thread(target=self.read_loop, daemon=True); self.th.start()

    def read_loop(self):
        # split the MJPEG stream at SOI (FFD8) / EOI (FFD9) / MJPEG 列を SOI/EOI で切り分けて保存
        fcsv = open(E.OUT / 'frames.csv', 'w', newline=''); w = csv.writer(fcsv)
        w.writerow(['index', 'wall_time_s', 'file'])
        buf = b''; fd = self.p.stdout.fileno()
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            t = time.time(); buf += chunk
            while True:
                s = buf.find(b'\xff\xd8')
                if s < 0:
                    buf = buf[-1:]; break
                e = buf.find(b'\xff\xd9', s + 2)
                if e < 0:
                    buf = buf[s:]; break
                self.n += 1
                name = f'img/{self.tag}_{self.n:06d}.jpg'
                (E.OUT / name).write_bytes(buf[s:e + 2]); buf = buf[e + 2:]
                w.writerow([self.n, f'{t:.2f}', name]); fcsv.flush()
        fcsv.close()

    def kill(self):
        if self.p and self.p.poll() is None:
            self.p.kill()

    def stop(self):
        # ask ffmpeg to quit ('q'), then terminate / kill if needed / 'q' で止め、だめなら terminate・kill
        if not self.p:
            return
        try:
            self.p.stdin.write(b'q'); self.p.stdin.flush()
        except Exception:
            pass
        for sig in (None, 'term', 'kill'):
            try:
                if sig == 'term':
                    self.p.terminate()
                elif sig == 'kill':
                    self.p.kill()
                self.p.wait(timeout=4); break
            except subprocess.TimeoutExpired:
                continue
        if self.th:
            self.th.join(timeout=3)
        self.ctl.event('camera_stop', f'frames={self.n}')


class Audio:
    """Background ffmpeg recording the microphone for the whole run to <tag>.wav (start time logged).
    運転の間ずっと録音して <tag>.wav に書く（開始時刻を記録）。"""

    def __init__(self, tag, ctl):
        self.path, self.ctl, self.p = E.OUT / f'{tag}.wav', ctl, None

    def start(self):
        mic = A.mic_index() if A.MIC_NAME else None
        if mic is None:
            self.ctl.event('audio_skip', 'SF_MIC_NAME not set or device not found'); return
        self.p = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'avfoundation',
                                   '-i', f':{mic}', '-ac', '1', '-ar', '48000', str(self.path)],
                                  stdin=subprocess.PIPE, start_new_session=True)
        atexit.register(lambda: self.p.poll() is None and self.p.kill())
        self.ctl.event('audio_start', str(self.path))

    def stop(self):
        if not self.p:
            return
        try:
            self.p.stdin.write(b'q'); self.p.stdin.flush(); self.p.wait(timeout=5)
        except Exception:
            self.p.kill()
        self.ctl.event('audio_stop', '')


# ---------------------------------------------------------------- common helpers
def add_args(ap):
    ap.add_argument('--motor', type=int, default=1)
    ap.add_argument('--fps', type=float, default=3.0, help='camera frames per second / カメラの毎秒枚数')
    ap.add_argument('--audio', action='store_true', help='record the microphone (needs SF_MIC_NAME) / 録音')
    ap.add_argument('--preroll-sec', type=float, default=8.0,
                    help='camera runs this long before the first command (start-up ~4.5 s + scale zero) / 最初の指令の前に撮る時間（起動約 4.5 秒＋秤のゼロ）')
    ap.add_argument('--postroll-sec', type=float, default=6.0,
                    help='camera keeps running this long after the final stop (scale settles ~2.5 s, then zero) / 最後の停止の後に撮る時間（秤が落ち着く約 2.5 秒＋ゼロ）')
    ap.add_argument('--dry-run', action='store_true', help='no scope, no serial: fake data, log commands only')
    ap.add_argument('--dry-run-camera', action='store_true', help='with --dry-run: really run the camera')
    E.add_common_args(ap)


def env_info():
    # environment values that affect the numbers / 数値に効く環境値
    return {'SF_CURRENT_SCALE': A.I_SCALE, 'SF_CAM_NAME': A.CAM, 'SF_MIC_NAME': A.MIC_NAME, 'PWM_FREQ_HZ': E.F_PWM,
            'CH2_DELAY_NS': E.SH, 'CH2_GAIN': E.G, 'CH2_OFFSET_V': E.O, 'N_ripples_per_rev': R.N,
            'SCOPE_IP': E.IP, 'SF_BRIDGE_DIR': str(A.BR)}


def start_media(a, ctl):
    # camera (and microphone) for the whole run / 運転の間ずっとカメラ（と録音）
    cam = Camera(a.tag, a.fps, ctl); aud = Audio(a.tag, ctl)
    if not a.dry_run or a.dry_run_camera:
        cam.start()
    if a.audio and not a.dry_run:
        aud.start()
    return cam, aud


def install_sigint(ctl, stop, motor_stop):
    # first Ctrl-C: stop flag + immediate motor stop; second: KeyboardInterrupt (abort a blocking readout)
    # 1 回目の Ctrl-C は停止フラグと即 motor stop、2 回目は KeyboardInterrupt（読み出し中でも抜ける）
    def handler(signum, frame):
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set(); ctl.event('stop_reason', 'SIGINT'); motor_stop('SIGINT')
    signal.signal(signal.SIGINT, handler)


# ---------------------------------------------------------------- cycle
POINT_COLS = ['wall_time_s', 'spin_elapsed_s', 'section', 'lap', 'step', 'duty_cmd', 'duty_meas', 'VDD_V', 'Vm_avg_V',
              'f_comm_Hz', 'omega_rad_s', 'Ibat_A', 'CLI_V', 'power_line', 'npz', 'send_time', 'trigger_time',
              'readout_s', 'skipped']


class CycleWorker(threading.Thread):
    """Capture / analyse / power for one step at a time; skips a capture that would start too late.
    1 段ずつ取り込み・解析・電力読みを行う。開始が遅れる取り込みは飛ばす（Duty の予定が優先）。"""

    def __init__(self, a, ctl, sc, stop):
        super().__init__(daemon=True)
        self.a, self.ctl, self.sc, self.stop = a, ctl, sc, stop
        self.q = queue.Queue()
        self.f = open(E.OUT / 'points.csv', 'w', newline=''); self.w = csv.DictWriter(self.f, POINT_COLS)
        self.w.writeheader(); self.f.flush()

    def run(self):
        while True:
            job = self.q.get()
            if job is None:
                break
            try:
                self.handle(job)
            except Exception as e:                 # a failed capture must not stop the duty schedule / 失敗しても Duty は続ける
                self.ctl.event('capture_error', repr(e))
                self.row(job, skipped=f'error:{e!r}')

    def row(self, job, **kw):
        r = {c: '' for c in POINT_COLS}
        r.update(wall_time_s=f'{time.time():.2f}', section=self.a.section, lap=job['lap'], step=job['step'],
                 duty_cmd=job['duty'], send_time=f"{job['send_time']:.2f}", skipped=0)
        r.update(kw)
        self.w.writerow(r); self.f.flush()

    def handle(self, job):
        a = self.a
        t_cap = job['t_start'] + a.wait_sec
        while time.time() < t_cap:
            if self.stop.wait(0.02):
                return
        if self.stop.is_set():
            return
        if time.time() - job['t_start'] > a.step_sec - 0.5:       # too late: keep the schedule, skip the capture
            self.ctl.event('capture_skipped', f"lap{job['lap']} step{job['step']}", 'late')
            self.row(job, skipped=1, spin_elapsed_s=f"{job['spin_t'] + time.time() - job['t_start']:.2f}")
            return
        sc = self.sc
        if a.dry_run:
            sc.duty = job['duty'] / 100
        sc.w(':SING'); t_trig = time.time()
        for k in range(100):                                      # wait for STOP, force a trigger after ~0.5 s
            if sc.q(':TRIG:STAT?') == 'STOP':
                break
            if k == 10:
                sc.w(':TFOR')
            time.sleep(0.05)
        # read the current right after the capture, before the readout, so it belongs to this step
        # 電流は取り込み直後（読み出しの前）に読む。読み出し中に次の段へ移っても、この段の値になる
        line = self.ctl.power()
        t0 = time.time()
        t, v1, _ = sc.grab_raw(1); _, v2, _ = sc.grab_raw(2); sc.w(':RUN')
        readout = time.time() - t0
        name = f"{a.tag}_L{job['lap']}_S{job['step']:02d}.npz"
        np.savez(E.OUT / name, t=t, v1=v1, v2=v2, mode='pwm')
        r, _ = E.analyse({'t': t, 'v1': v1, 'v2': v2}, 'pwm')
        avg, vdd, vm = R.per_period_mean(t, v1, v2)
        try:
            f_comm = R.fine_peak(avg, E.F_PWM, r['f_comm_Hz'])
        except Exception:
            f_comm = r['f_comm_Hz']                               # fall back to the coarse FFT peak / 粗い FFT の値
        pw = A.parse_power(line) if line else None
        self.row(job, spin_elapsed_s=f"{job['spin_t'] + t_trig - job['t_start']:.2f}", duty_meas=f'{np.mean(v1 < 1.0):.5f}',
                 VDD_V=f'{vdd:.4f}', Vm_avg_V=f'{vm:.4f}', f_comm_Hz=f'{f_comm:.2f}',
                 omega_rad_s=f'{2 * np.pi * f_comm / R.N:.2f}', Ibat_A=f'{pw[1]:.4f}' if pw else '',
                 CLI_V=f'{pw[0]:.3f}' if pw else '', power_line=line or '', npz=name, trigger_time=f'{t_trig:.2f}',
                 readout_s=f'{readout:.2f}')
        self.ctl.event('capture', f"lap{job['lap']} step{job['step']}", f'readout={readout:.2f}s VDD={vdd:.3f}')
        if vdd < a.vmin and not self.stop.is_set():
            self.ctl.event('stop_reason', f'VDD {vdd:.3f} V < vmin {a.vmin} V')
            self.stop.set()


def sleep_until(t, stop):
    # sleep until wall time t or until stop is set / 時刻 t まで待つ（停止で中断）
    while not stop.is_set():
        dt = t - time.time()
        if dt <= 0:
            return
        stop.wait(min(dt, 0.1))


def rest_power(ctl, note):
    # current with the motor stopped: sensor offset or battery-side load, subtracted from the spinning readings later
    # モータ停止中の電流（センサのゼロのずれか電池側の消費）。後で回転中の読みから引く
    ctl.event('rest_power', ctl.power() or '', note)


def run_cycle(a, ctl, sc, stop):
    # duty laps: the schedule is driven here, captures run in the worker / 周回の予定はここで進め、取り込みは worker
    worker = CycleWorker(a, ctl, sc, stop); worker.start()
    spin = 0.0; lap = 0
    rest_power(ctl, 'before run')
    while lap < a.laps and spin < a.total_sec and not stop.is_set():
        lap += 1; ctl.event('lap_start', str(lap))
        t_base = time.time()
        for i, duty in enumerate(a.duty_list, 1):
            if stop.is_set() or spin >= a.total_sec:
                break
            t_start = max(time.time(), t_base)
            sleep_until(t_start, stop)
            t_send = ctl.send(f'motor test {a.motor} {duty} {a.step_sec + 2:g}', f'lap{lap} step{i}')
            worker.q.put(dict(lap=lap, step=i, duty=duty, t_start=t_send, send_time=t_send, spin_t=spin))
            t_base = t_start + a.step_sec
            sleep_until(t_base, stop)
            if not stop.is_set():
                spin += a.step_sec
        if stop.is_set():
            break
        if lap < a.laps and spin < a.total_sec and a.zero_stop_sec > 0:      # zero window between laps / 周の間のゼロ窓
            ctl.send('motor stop', 'zero window'); t0 = time.time()
            sleep_until(t0 + min(2.5, a.zero_stop_sec / 2), stop)
            if not stop.is_set():
                rest_power(ctl, f'zero window after lap {lap}')
            sleep_until(t0 + a.zero_stop_sec, stop)
            ctl.event('zero_window', f'start={t0:.2f} end={time.time():.2f}')
    if not stop.is_set():
        ctl.event('stop_reason', 'laps done' if lap >= a.laps else 'total-sec reached')
    worker.q.put(None); worker.join(timeout=60)


# ---------------------------------------------------------------- step
STEP_COLS = ['pair', 'pre', 'post', 'pre_send_time', 'trigger_time', 'post_send_time', 'readout_s',
             'power_before', 'power_after', 'npz']


def on_width(pct):
    # drain low (MOSFET on) time per PWM period for a commanded duty %, after the firmware's 8-bit truncation
    # 指令 Duty % のときの 1 周期あたりのドレインの Low（MOSFET ON）時間。ファームの 8 bit 切り捨て込み
    return (int(pct / 100 * 255) / 255) / E.F_PWM


def arm_step_trigger(sc, pre, post, level=0.45):
    # make the duty step itself trigger the scope (no PC timing involved); level 0.45 V works with and without
    # the propeller (drain ~0 V while on, >= ~0.9 V while off)
    # 段差そのものでオシロをトリガさせる（PC のタイミングに頼らない）。0.45 V はプロペラの有無によらず使える
    # （ON 中のドレインは約 0 V、OFF 中は約 0.9 V 以上）
    if pre == 0:                                   # first on-pulse of the spin-up / 回し始めの最初の ON パルス
        cmds = [':TRIG:MODE EDGE', ':TRIG:EDG:SOUR CHAN1', ':TRIG:EDG:SLOP NEG', f':TRIG:EDG:LEV {level}']
    elif post == 0:                                # spin-down: trigger on the running PWM, stop 0.4 s later (as exp_coast)
        # スピンダウン: 回転中の PWM でトリガし、0.4 秒後に止める（exp_coast と同じ。2026-11-02 に動作確認）
        cmds = [':TRIG:MODE EDGE', ':TRIG:EDG:SOUR CHAN1', ':TRIG:EDG:SLOP NEG', f':TRIG:EDG:LEV {level}']
    else:
        mid = (on_width(pre) + on_width(post)) / 2
        if post > pre:                             # first on-pulse longer than the midpoint / 中間より長い最初の ON パルス
            cmds = [':TRIG:MODE PULS', ':TRIG:PULS:SOUR CHAN1', ':TRIG:PULS:WHEN NGR', f':TRIG:PULS:WIDT {mid:.4e}']
        else:                                      # shorter than the midpoint, but not a ringing glitch / 中間より短く、ノイズより長い
            # upper limit first: the scope requires lower < upper at every moment / 上限を先に（常に下限 < 上限が必要）
            cmds = [':TRIG:MODE PULS', ':TRIG:PULS:SOUR CHAN1', ':TRIG:PULS:WHEN NGL',
                    f':TRIG:PULS:UWID {mid:.4e}', f':TRIG:PULS:LWID {on_width(post) / 2:.4e}']
        cmds.append(f':TRIG:PULS:LEV {level}')
    for c in cmds:
        sc.w(c)
    time.sleep(0.2)
    return ' '.join(cmds)


def run_steps(a, ctl, sc, stop):
    # duty steps: pre duty -> single long capture -> force trigger -> post duty -> read out / 段差 1 つずつ
    fcsv = open(E.OUT / 'steps.csv', 'w', newline=''); w = csv.DictWriter(fcsv, STEP_COLS); w.writeheader()
    state = {'hold': None, 'last': 0.0}

    def keepalive():
        # re-send the same short-expiry motor test every keepalive/2 s while a duty is held, so the motor stops
        # within ~keepalive s if the PC side or the link (WiFi) dies
        # 保持中の Duty を keepalive/2 秒ごとに短い期限で再送。PC や通信（WiFi）が落ちても約 keepalive 秒で止まる
        while not stop.wait(0.25):
            with ctl.lock:
                if state['hold'] and not stop.is_set() and time.time() - state['last'] >= a.keepalive_sec / 2:
                    state['last'] = ctl.send(f"motor test {a.motor} {state['hold']} {a.keepalive_sec:g}", 'keepalive')
    threading.Thread(target=keepalive, daemon=True).start()

    def set_duty(d, note):
        # motor stop for 0, else motor test with the short expiry; returns send time / 0 なら motor stop、他は短い期限の motor test
        with ctl.lock:
            state['hold'] = d if d else None
            t = ctl.send('motor stop' if d == 0 else f'motor test {a.motor} {d} {a.keepalive_sec:g}', note)
            state['last'] = t
        return t

    # scope setup as exp_coast: falling edge on CH1, single, trigger near the left edge / exp_coast と同じ設定
    sc.w(':RUN')
    sc.w(':CHAN1:SCAL 1.25'); sc.w(':CHAN1:OFFS -3.25'); sc.w(':CHAN2:SCAL 0.5'); sc.w(':CHAN2:OFFS -3.5')
    sc.w(':TRIG:SWE NORM')
    # trigger 1.5 div from the left edge: the record spans -1.5 div .. +10.5 div around it
    # トリガを左端から 1.5 目盛りに置く。記録はその前 1.5 目盛り、後 10.5 目盛り
    sc.w(f':TIM:MAIN:SCAL {a.tb}'); sc.w(f':TIM:MAIN:OFFS {4.5 * a.tb}')
    time.sleep(0.5); sc.w(f':ACQ:MDEP {a.mdep}'); time.sleep(1.0)
    for k, (pre, post) in enumerate(a.step_list, 1):
        if stop.is_set():
            break
        row = dict(pair=k, pre=pre, post=post)
        row['pre_send_time'] = f'{set_duty(pre, f"pair{k} pre"):.2f}'
        sleep_until(time.time() + a.pre_sec, stop)
        if stop.is_set():
            break
        row['power_before'] = ctl.power() or ''
        if a.dry_run:
            sc.step = (pre / 100, post / 100)
        # arming over LAN takes 0.1-0.2 s or more: set a trigger that only the step can satisfy, wait until the
        # scope is armed (WAIT), then send the step; force a trigger only as a fallback after 3 s
        # LAN 越しの待ち受け開始は 0.1〜0.2 秒以上かかる: 段差でしか成り立たないトリガを設定し、待ち受け（WAIT）を
        # 確かめてから段差を送る。3 秒たっても待ち受けにならなければ強制トリガ
        trig = arm_step_trigger(sc, pre, post)
        ctl.event('trigger_setup', trig, f'pair{k}')
        sc.w(':SING'); t_sing = time.time(); scope_st = ''
        if pre > 0 and post == 0:                  # spin-down: the running PWM triggers at once / 回転中の PWM ですぐトリガ
            time.sleep(0.4); scope_st = 'spin-down 0.4 s'
        else:
            while time.time() - t_sing < 3.0:      # armed and waiting: the step will trigger it / 待ち受け中なら段差でトリガする
                scope_st = sc.q(':TRIG:STAT?')
                if scope_st in ('WAIT', 'TD', 'STOP'):
                    break
                time.sleep(0.02)
            else:
                sc.w(':TFOR'); scope_st = 'forced'
        row['trigger_time'] = f'{time.time():.2f}'
        ctl.event('scope_state', scope_st, f'pair{k} {time.time() - t_sing:.2f}s after SING')
        row['post_send_time'] = f'{set_duty(post, f"pair{k} post"):.2f}'
        t0 = time.time()
        for _ in range(int((a.tb * 12 + 20) / 0.1)):             # wait for STOP / STOP を待つ
            if sc.q(':TRIG:STAT?') == 'STOP':
                break
            time.sleep(0.1)
        row['power_after'] = ctl.power() or ''                   # still at the post duty / まだ段差後の Duty
        if not a.run_during_readout and post != 0:
            set_duty(0, f'pair{k} readout stop')                 # stop during the long readout / 長い読み出しの間は止める
        t1 = time.time()
        t, v1, _ = sc.grab_raw(1); _, v2, _ = sc.grab_raw(2); sc.w(':RUN')
        row['readout_s'] = f'{time.time() - t1:.2f}'
        name = f'{a.tag}_{pre}to{post}.npz'
        np.savez(E.OUT / name, t=t, v1=v1, v2=v2, mode='step', pre=pre, post=post); row['npz'] = name
        w.writerow(row); fcsv.flush()
        ctl.event('step_done', f'pair{k} {pre}to{post}', f"readout={row['readout_s']}s")
    # leave an ordinary edge trigger behind for the next script / 次のスクリプトのため普通のエッジトリガに戻す
    sc.w(':TRIG:MODE EDGE'); sc.w(':TRIG:EDG:SOUR CHAN1'); sc.w(':TRIG:EDG:SLOP NEG'); sc.w(':TRIG:EDG:LEV 2.0')
    sc.w(':TRIG:SWE AUTO'); sc.w(':RUN')
    fcsv.close()


# ---------------------------------------------------------------- main
def parse_int_list(s):
    return [int(x) for x in s.split(',') if x.strip() != '']


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cycle'); c.add_argument('tag')
    c.add_argument('--duties', default='50,60,62,64,66,68,70,72,74,76,78,80')
    c.add_argument('--step-sec', type=float, default=5.0); c.add_argument('--laps', type=int, default=1)
    c.add_argument('--total-sec', type=float, default=240.0); c.add_argument('--zero-stop-sec', type=float, default=4.0)
    c.add_argument('--wait-sec', type=float, default=1.5); c.add_argument('--vmin', type=float, default=3.5)
    # 60k: capture + readout about 3.1 s on the DS1054Z (600k took 7.4 s, too long for a 5-s step; measured 2026-11-02)
    # 60k: 取り込みと読み出しで約 3.1 秒（600k は 7.4 秒で 5 秒の段に収まらない。2026-11-02 実測）
    c.add_argument('--mdep', type=int, default=60000, help='scope memory depth for the PWM capture / PWM 取り込みのメモリ長')
    c.add_argument('--section', default='A', choices=['A', 'B', 'C'])
    add_args(c)
    s = sub.add_parser('step'); s.add_argument('tag')
    s.add_argument('--steps', default='0:60,60:68,68:60,50:80,80:50,60:0')
    s.add_argument('--pre-sec', type=float, default=2.5); s.add_argument('--tb', type=float, default=0.1)
    # 6M at 0.1 s/div = 5 MS/s, as the 10-15 spin-downs; the readout takes about 48 s
    # 6M・0.1 s/div で 5 MS/s（10-15 のスピンダウンと同じ）。読み出しは約 48 秒
    s.add_argument('--mdep', type=int, default=6000000)
    s.add_argument('--run-during-readout', action='store_true',
                   help='keep the post duty during the ~48 s readout (default: stop the motor to limit heating and battery drain) / '
                        '読み出し中も段差後の Duty で回す（既定は止めて、発熱と電池の消耗を抑える）')
    s.add_argument('--keepalive-sec', type=float, default=10.0,
                   help='motor test expiry; re-sent every half of it / motor test の期限（半分ごとに再送）')
    add_args(s)
    a = ap.parse_args()
    E.apply_common_args(a, need_scope=not a.dry_run)
    # the current scale differs between firmware versions (10 before the 0.01-ohm shunt fix, 1 after): require it explicitly
    # 電流の倍率は機体ソフトの版で違う（0.01 Ω の修正前は 10、後は 1）ので、明示を必須にする
    if not a.dry_run and 'SF_CURRENT_SCALE' not in os.environ:
        raise SystemExit('set SF_CURRENT_SCALE explicitly (1 for firmware with the 0.01 ohm shunt fix) / SF_CURRENT_SCALE を明示してください')
    E.check_tag(a.tag)
    if not 1 <= a.motor <= 4:
        raise SystemExit('motor must be 1-4')
    if a.cmd == 'cycle':
        a.duty_list = parse_int_list(a.duties)
        if not a.duty_list or not all(0 <= d <= 100 for d in a.duty_list):
            raise SystemExit('duties must be integers 0-100')       # firmware accepts integer % only / 整数 % のみ
    else:
        a.step_list = [tuple(int(x) for x in p.split(':')) for p in a.steps.split(',')]
        if not all(0 <= d <= 100 for p in a.step_list for d in p):
            raise SystemExit('step duties must be integers 0-100')
    ctl = Ctl(a.dry_run); stop = threading.Event()
    sc = FakeScope(a.mdep) if a.dry_run else E.Scope()
    info = {'args': {k: v for k, v in vars(a).items() if k != 'step_list'}, 'env': env_info(),
            'start': time.strftime('%Y-%m-%d %H:%M:%S'), 'start_wall_s': round(time.time(), 2)}
    run_json = E.OUT / 'run.json'
    run_json.write_text(json.dumps(info, indent=1, ensure_ascii=False, default=str))
    cam, aud = start_media(a, ctl)

    def motor_stop(why):
        try:
            ctl.send('motor stop', why)
        except Exception as e:
            print('motor stop failed:', e)
    install_sigint(ctl, stop, motor_stop)
    # pre-roll: let the camera start and record the scale zero before the motor turns
    # 前の撮影: カメラの起動を待ち、モータを回す前の秤のゼロを撮る
    if cam.p is not None and a.preroll_sec > 0:
        ctl.event('preroll', f'{a.preroll_sec:g}s')
        sleep_until(time.time() + a.preroll_sec, stop)
    try:
        if a.cmd == 'cycle':
            if not a.dry_run:
                E.setup(sc, 'pwm'); sc.w(':TRIG:MODE EDGE'); sc.w(':TRIG:EDG:SOUR CHAN1'); sc.w(':TRIG:EDG:SLOP NEG')
                sc.w(':TRIG:EDG:LEV 2.0'); sc.w(f':ACQ:MDEP {a.mdep}'); time.sleep(1.5)
                try:
                    info['scope_srate'] = sc.q(':ACQ:SRAT?')
                except Exception:
                    pass
            ctl.event('run_start', a.cmd)
            run_cycle(a, ctl, sc, stop)
        else:
            ctl.event('run_start', a.cmd)
            run_steps(a, ctl, sc, stop)
    except KeyboardInterrupt:
        ctl.event('stop_reason', 'KeyboardInterrupt')
    except Exception as e:
        ctl.event('stop_reason', f'exception {e!r}'); print('error:', repr(e))
    finally:
        stop.set()
        motor_stop('final (finally)')                                # always stop the motor first / 最優先でモータを止める
        # post-roll: keep the camera on until the scale has settled and shows the zero after the run
        # 後の撮影: 秤が落ち着いて運転後のゼロを示すまでカメラを止めない
        if cam.p is not None and a.postroll_sec > 0:
            ctl.event('postroll', f'{a.postroll_sec:g}s')
            try:
                time.sleep(min(2.5, a.postroll_sec))
                rest_power(ctl, 'after run')
                time.sleep(max(0.0, a.postroll_sec - 2.5))
            except KeyboardInterrupt:
                pass
        cam.stop(); aud.stop()
        info['end'] = time.strftime('%Y-%m-%d %H:%M:%S'); info['frames'] = cam.n
        run_json.write_text(json.dumps(info, indent=1, ensure_ascii=False, default=str))
        motor_stop('after run')                                      # and once more at the very end / 最後にもう 1 回
    print('done:', E.OUT)


if __name__ == '__main__':
    main()
