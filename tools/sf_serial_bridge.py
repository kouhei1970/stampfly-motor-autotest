#!/usr/bin/env python3
"""Hold the StampFly USB serial port open (opening it reboots the board) and bridge it.
StampFly の USB シリアルを開いたまま保持する（開くたびに再起動するため）中継。
Log: <dir>/serial.log (all output), commands: write a line to <dir>/cmd.fifo (named pipe).
The folder, FIFO and log are private to the user (0700 / 0600): only the owner can send commands.
フォルダ・FIFO・ログは本人だけが読み書きできる（0700 / 0600）。コマンドを送れるのは本人だけ。
Usage: python3 sf_serial_bridge.py [dir] [port]   (port default: the single /dev/cu.usbmodem*)   (default dir: $SF_BRIDGE_DIR or ~/.cache/sf_serial_bridge)
"""
import glob, os, pathlib, stat, sys, threading, time, serial

d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                 os.environ.get('SF_BRIDGE_DIR', str(pathlib.Path.home() / '.cache' / 'sf_serial_bridge')))
# port: argument, else the only /dev/cu.usbmodem* (its number changes with the USB socket)
# ポート：引数、なければ唯一の /dev/cu.usbmodem*（番号は USB の差し口で変わる）
ports = sorted(glob.glob('/dev/cu.usbmodem*'))
port = sys.argv[2] if len(sys.argv) > 2 else (ports[0] if len(ports) == 1 else None)
if port is None:
    raise SystemExit(f'give the port explicitly; found {ports}')
d.mkdir(parents=True, exist_ok=True); os.chmod(d, 0o700)
fifo = d / 'cmd.fifo'
if fifo.exists() and not stat.S_ISFIFO(fifo.stat().st_mode):
    raise SystemExit(f'{fifo} exists and is not a FIFO')
if not fifo.exists():
    os.mkfifo(fifo, 0o600)
os.chmod(fifo, 0o600)
s = serial.Serial(port, 115200, timeout=0.2)
logp = d / 'serial.log'
fd = os.open(logp, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
log = os.fdopen(fd, 'ab', buffering=0)


def reader():
    while True:
        b = s.read(4096)
        if b:
            log.write(b)


threading.Thread(target=reader, daemon=True).start()
while True:
    with open(fifo) as f:              # blocks until a writer opens / 書き手が開くまで待つ
        for line in f:
            line = line.strip()[:200]  # one CLI line, bounded length / 1 行、長さに上限
            if not line:
                continue
            log.write(f'\n[{time.strftime("%H:%M:%S")} SEND] {line}\n'.encode())
            s.write((line + '\r\n').encode())
