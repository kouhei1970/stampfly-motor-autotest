#!/bin/bash
# Run PWM steps D0.1..0.9 (+ D0.5 repeat) on one motor; stop if battery is low.
# D0.1〜0.9（最後に D0.5 の再現）を順に実行。電池が低ければ止める。
# Usage: tools/exp_batch.sh <motor> <cap tag e.g. c01> [VMIN=3.6]
# Env: AUTOTEST_DATA (output folder, default ./data), SCOPE_IP (oscilloscope), SF_BRIDGE_DIR,
#      DUTIES (e.g. "10 20 30 40 50 60" for no-prop), AUDIO=--audio (needs SF_MIC_NAME)
# 環境変数: AUTOTEST_DATA（出力先、既定 ./data）、SCOPE_IP（オシロ）、DUTIES（プロペラなしは 0.6 まで）、AUDIO=--audio（録音）
M=$1; C=$2; VMIN=${3:-3.6}; AUDIO=${AUDIO-}
[ -n "$M" ] && [ -n "$C" ] || { echo "usage: $0 <motor 1-4> <cap tag> [VMIN]"; exit 1; }
[ -n "$SCOPE_IP" ] || { echo "SCOPE_IP is not set (oscilloscope IP) / SCOPE_IP が未設定です"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA=${AUTOTEST_DATA:-./data}
mkdir -p "$DATA"
DUTIES=${DUTIES:-"10 20 30 40 50 60 70 80 90 50"}
for D in $DUTIES; do
  TAG=pwm_${C}_d$(printf %02d $D); [ -f "$DATA/$TAG.npz" ] && TAG=${TAG}_rep
  python3 "$HERE/exp_auto.py" "$TAG" "$M" "$D" --sec 14 --data "$DATA" $AUDIO || exit 1
  V=$(python3 -c 'import json,sys;r=json.load(open(sys.argv[2]+"/results.json"))[sys.argv[1]];print(r["power_after"].split()[1] if r["power_after"] else 0)' "$TAG" "$DATA")
  echo "== $TAG done, battery after = $V V"
  awk "BEGIN{exit !($V < $VMIN)}" && { echo "LOW BATTERY ($V V) - stop"; exit 2; }
  sleep 10   # cool-down / 休止
done
