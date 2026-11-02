#!/usr/bin/env python3
"""Reads the seven-segment LCD of the scale from camera photos (numpy + PIL only).
カメラ写真から秤の 7 セグメント LCD を読む（numpy と PIL のみ）。

Usage: python3 tools/read_scale.py <run folder> [--calib FILE]
       python3 tools/read_scale.py --validate tools/scale_validation_2026-11-02.csv
  <run folder> holds img/ (exp_long run: frames.csv + single frames; exp_auto: tiles of 3 or 9 frames)
  or holds several such run folders. Writes <run folder>/scale_readings.csv:
  file, frame_in_tile, wall_time_s, value_g, ok, detail
Method / 方法:
  1. Ink map: darkness = local background - gray, normalised by the ink level, so lighting does not matter.
     インク図：局所背景 - 輝度。インク濃度で正規化するので照明の変化に影響されない。
  2. The ink bounding box gives a first guess of the digit block position and size (the camera moved
     between sessions); a grid search over shift and scale then fits 4 digit cells (tens, ones, tenths,
     hundredths) to the picture.  Each cell is 7 segment sample lines that follow the italic slant.
     インクの外接矩形から位置と大きさの初期値を作り、平行移動と拡大率の格子探索で 4 桁のセルを合わせる。
  3. A segment is on / off by its normalised darkness; a frame is ok only when every segment is clearly on
     or clearly off, every digit is a valid pattern, the decimal point is on, and no stray ink lies outside
     the predicted segments (doubled or fading segments while the LCD updates are rejected).
     全セグメントが明確に点灯／消灯、全桁が有効な字形、小数点が点灯、予測位置の外にインクが無いときだけ ok。
"""
import argparse, csv, json, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw

CALIB_DEFAULT = pathlib.Path(__file__).with_name('scale_calib_2026-11-02.json')
SEGS = 'abcdefg'
PATTERNS = {   # digit -> lit segments (7 has two usual variants) / 各数字の点灯セグメント
    '0': 'abcdef', '1': 'bc', '2': 'abdeg', '3': 'abcdg', '4': 'bcfg', '5': 'acdfg',
    '6': 'acdefg', '7': 'abc', '8': 'abcdefg', '9': 'abcdfg'}
EXTRA = [('7', 'abcf')]
CELLS = 4            # tens, ones, tenths, hundredths / 十の位、一の位、小数第 1 位、第 2 位
DP_AFTER_CELL = 1    # decimal point sits right of the ones digit / 小数点は一の位の右


def box_mean(a, k):
    # box filter by cumulative sums / 累積和による箱形平滑
    p = np.pad(a, ((k // 2 + 1, k // 2), (k // 2 + 1, k // 2)), mode='edge')
    c = p.cumsum(0).cumsum(1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def template_table():
    # (n_templates, 7) 0/1 table and the digit label of each row / 字形表とラベル
    rows, labels = [], []
    for d, s in list(PATTERNS.items()) + EXTRA:
        rows.append([1.0 if g in s else 0.0 for g in SEGS]); labels.append(d)
    return np.array(rows), labels


def darkness_map(gray, cal):
    # local-background-relative darkness (positive = ink); two passes so that a brightness gradient
    # (vignetting at the picture edge) is followed by the background estimate
    # 局所背景に対する暗さ（正がインク）。明るさの勾配（画像端の周辺減光）に背景推定が追従するよう 2 段で求める
    rough = box_mean(gray, cal['bg_box'] + 20)
    bg = box_mean(np.maximum(gray, rough), cal['bg_box'])
    return np.clip(bg - gray, 0, None)


def find_block(dark, cal):
    # first guess of the digit block: right edge, top, bottom, ink level / 数字ブロックの右端・上端・下端・インク濃度
    r = cal['region']
    sub = dark[r['y0']:r['y1'], r['x0']:r['x1']]
    level = float(np.percentile(sub, 99.5))
    if level < cal['min_ink_level']:
        return None, 'no ink (contrast %.0f)' % level
    mask = sub > cal['ink_mask_frac'] * level
    cols = np.nonzero(mask.sum(0) >= cal['min_col_px'])[0]
    if cols.size == 0:
        return None, 'no ink columns'
    # split into runs of columns; the small "g" unit mark at the right edge is dropped
    # 列の連なりに分け、右端の小さな単位記号 g は除く
    runs, start = [], cols[0]
    for a, b in zip(cols[:-1], cols[1:]):
        if b - a > cal['run_gap_px']:
            runs.append((start, a)); start = b
    runs.append((start, cols[-1]))
    spans = []
    for x0, x1 in runs:
        rows = np.nonzero(mask[:, x0:x1 + 1].sum(1) >= 1)[0]
        spans.append((x0, x1, rows.min(), rows.max()))
    if len(spans) > 1 and spans[-1][3] - spans[-1][2] < cal['g_max_height_px'] and spans[-1][0] > cal['g_min_x'] - r['x0']:
        spans = spans[:-1]
    x_left = min(s[0] for s in spans); x_right = max(s[1] for s in spans)
    keep = mask[:, x_left:x_right + 1]
    rowsum = keep.sum(1)
    rows = np.nonzero(rowsum >= cal['min_row_px'])[0]
    if rows.size == 0:
        return None, 'no ink rows'
    # keep the run of rows with the most ink: the digits, not a shadow band at the LCD edge
    # インクが最も多い連続行（数字）を採用し、LCD 縁の影の帯は除く
    cuts = np.nonzero(np.diff(rows) > cal['run_gap_px'])[0] + 1
    runs_r = np.split(rows, cuts)
    rows = max(runs_r, key=lambda rr: rowsum[rr].sum())
    return dict(right=x_right + r['x0'], top=rows.min() + r['y0'], bottom=rows.max() + r['y0'],
                left=x_left + r['x0'], level=level), ''


def seg_lines(cal):
    # sample points of 7 segments in cell coordinates (u right, v down, origin cell centre, no slant)
    # セル座標（u 右、v 下、原点はセル中心、傾き無し）での 7 セグメントの標本点
    g = cal['geometry']; hw, hh, m, n = g['half_width'], g['half_height'], g['seg_inset'], g['points_per_seg']
    t = np.linspace(-1, 1, n)
    horiz = lambda v: np.stack([t * (hw - m), np.full(n, v)], 1)
    vert = lambda u, v0, v1: np.stack([np.full(n, u), np.linspace(v0, v1, n)], 1)
    return np.stack([horiz(-hh), vert(hw, -hh + m, -m), vert(hw, m, hh - m), horiz(hh),
                     vert(-hw, m, hh - m), vert(-hw, -hh + m, -m), horiz(0.0)])   # (7, n, 2)


def sample_points(cal, p):
    # pixel coordinates of every sample point for a batch of alignments p (dict of equal-length arrays:
    # xc3, yc, s, slant, pitch, tilt) / 配置の束 p（xc3, yc, s, slant, pitch, tilt の配列）での全標本点の画素座標
    uv = seg_lines(cal)                                         # (7, n, 2)
    col = lambda k: np.asarray(p[k], float)[..., None, None, None]
    cells = (np.arange(CELLS) - (CELLS - 1))[:, None, None]     # cell index relative to the last one
    u = cells * col('pitch') + uv[None, :, :, 0]                # (A, 4, 7, n)
    # tilt: the camera is slightly rotated, so cells further left sit lower by tilt px per cell pitch
    # 傾き：カメラがわずかに回転しているので、左のセルほど tilt px（セル 1 つあたり）だけ低い
    v = uv[None, :, :, 1] + col('tilt') * (-cells)
    x = col('xc3') + col('s') * (u - col('slant') * v)
    y = col('yc') + col('s') * v
    return x, y


def dp_point(cal, p):
    # pixel position of the decimal point / 小数点の画素位置
    g = cal['geometry']
    arr = lambda k: np.asarray(p[k], float)
    u = (DP_AFTER_CELL + 0.5 - (CELLS - 1)) * arr('pitch') + g['dp_u']
    v = g['dp_v'] + arr('tilt') * (CELLS - 1 - DP_AFTER_CELL - 0.5)
    return arr('xc3') + arr('s') * (u - arr('slant') * v), arr('yc') + arr('s') * v


def sample(img, x, y):
    # nearest-pixel lookup with clipping / 最近傍画素の参照（範囲外は端に丸める）
    h, w = img.shape
    return img[np.clip(np.rint(y).astype(int), 0, h - 1), np.clip(np.rint(x).astype(int), 0, w - 1)]


def seg_gain(g):
    # typical darkness of a lit segment a..g (some segments are dimmer in this camera view); 1 if not calibrated
    # 点灯セグメント a..g の標準的な濃さ（このカメラ視点では暗めに写るセグメントがある）。未較正なら 1
    return np.array(g.get('seg_gain', [1.0] * 7))


PARAMS = ['xc3', 'yc', 's', 'slant', 'pitch', 'tilt']


def misfit(norm, cal, p, tabs):
    # how badly each alignment in the batch p explains the picture (sum of squared segment errors + decimal point)
    # 配置の束 p が画像をどれだけ説明できないか（セグメント誤差の二乗和＋小数点）
    x, y = sample_points(cal, p)
    lev = sample(norm, x, y).mean(-1) / seg_gain(cal['geometry'])   # (A, 4, 7)
    cost = np.zeros(lev.shape[0])
    for c in range(CELLS):
        cost += ((lev[:, c, None, :] - tabs[c][None]) ** 2).sum(-1).min(-1)
    dpx, dpy = dp_point(cal, p)
    return cost + (1.0 - sample(norm, dpx, dpy)) ** 2


def pick_spread(cost, p, keep, min_dx=8.0, min_dy=6.0):
    # best `keep` alignments that differ from each other in position (so one wrong basin cannot fill the list)
    # 位置が互いに離れた上位 keep 個（誤った谷 1 つで候補が埋まらないように）
    chosen = []
    for k in np.argsort(cost):
        if all(abs(p['xc3'][k] - p['xc3'][j]) >= min_dx or abs(p['yc'][k] - p['yc'][j]) >= min_dy for j in chosen):
            chosen.append(k)
            if len(chosen) == keep:
                break
    return chosen


def fit(norm, block, cal):
    # coarse grid search of shift and scale from each starting shape, then coordinate descent on all alignment
    # parameters; returns the alignment with the best misfit + unexplained-ink score
    # 各出発形状から平行移動と拡大率を粗く格子探索し、全パラメータを座標降下で追い込む。
    # 不一致度＋説明できないインクが最小の配置を返す
    g = cal['geometry']; tab, _ = template_table()
    tabs = [np.vstack([tab, np.zeros((1, 7))])] + [tab] * (CELLS - 1)   # the tens cell may be blank / 十の位は空白可
    s0 = (block['bottom'] - block['top']) / (2 * g['half_height'] + g['stroke'])
    yc0 = (block['top'] + block['bottom']) / 2.0
    sr = cal['search']
    best = None
    for st in cal['starts']:
        xc0 = block['right'] - s0 * (g['half_width'] + st['slant'] * g['half_height'] + g['stroke'] / 2)
        D, Y, S = np.meshgrid(np.arange(-sr['dx'], sr['dx'] + 1, 2.0), np.arange(-sr['dy'], sr['dy'] + 1, 2.0),
                              s0 * np.arange(1 - sr['scale'], 1 + sr['scale'] + 1e-9, 0.025), indexing='ij')
        n = D.size
        p = dict(xc3=xc0 + D.ravel(), yc=yc0 + Y.ravel(), s=S.ravel(), slant=np.full(n, st['slant']),
                 pitch=np.full(n, st['pitch']), tilt=np.full(n, st['tilt']))
        for k in pick_spread(misfit(norm, cal, p, tabs), p, sr['keep']):
            q = {name: float(p[name][k]) for name in PARAMS}
            steps = dict(xc3=1.0, yc=1.0, s=0.01, slant=0.02, pitch=1.0, tilt=1.0)
            for _ in range(3):                                  # coordinate descent, steps halve each round
                for name in PARAMS:
                    trial = {nm: np.full(3, v) for nm, v in q.items()}
                    trial[name] = q[name] + steps[name] * np.array([-1.0, 0.0, 1.0])
                    q[name] = float(trial[name][np.argmin(misfit(norm, cal, trial, tabs))])
                steps = {nm: v / 2 for nm, v in steps.items()}
            al = dict(q, cost=float(misfit(norm, cal, {nm: np.array([v]) for nm, v in q.items()}, tabs)[0]),
                      start=st['name'])
            _, _, on, _, dp = decode(norm, al, cal)
            al['stray'] = stray_fraction(norm, al, on, dp, cal)[0]
            al['score'] = al['cost'] + sr['stray_weight'] * al['stray']
            if best is None or al['score'] < best['score']:
                best = al
    return best


def decode(norm, align, cal):
    # segment levels -> digits, with confidence checks / セグメント濃度から数字へ（確信度の検査付き）
    th = cal['thresholds']; tab, labels = template_table()
    x, y = sample_points(cal, {k: np.array([v]) for k, v in align.items() if k in PARAMS})
    lev = (sample(norm, x, y).mean(-1) / seg_gain(cal['geometry']))[0]   # (4, 7)
    on = lev > th['on']
    digits, reasons = [], []
    for c in range(CELLS):
        ambiguous = ((lev[c] > th['off_max']) & (lev[c] < th['on_min'])).any()
        match = [i for i in range(len(tab)) if np.array_equal(tab[i] > 0.5, on[c])]
        if ambiguous:
            reasons.append('cell%d unclear segment (%s)' % (c, ' '.join('%.2f' % v for v in lev[c])))
            digits.append('?')
        elif not on[c].any():
            digits.append(' ')
        elif match:
            digits.append(labels[match[0]])
        else:
            reasons.append('cell%d invalid pattern %s' % (c, ''.join(sg for sg, o in zip(SEGS, on[c]) if o)))
            digits.append('?')
    if digits[1] == ' ' or digits[2] == ' ' or digits[3] == ' ':
        reasons.append('blank in ones/tenths/hundredths')
    dpx, dpy = dp_point(cal, {k: np.array([v]) for k, v in align.items() if k in PARAMS})
    dp_level = float(sample(norm, dpx, dpy)[0])
    if dp_level < th['dp_on']:
        reasons.append('decimal point not seen (%.2f)' % dp_level)
    return digits, lev, on, reasons, (float(dpx[0]), float(dpy[0]))


def stray_fraction(norm, align, on, dp, cal, ink_level=None):
    # share of ink outside the predicted segments: doubled / fading segments show up here
    # 予測セグメントの外にあるインクの割合。二重像や消えかけのセグメントがここに出る
    g = cal['geometry']; h, w = norm.shape
    x, y = sample_points(cal, {k: np.array([v]) for k, v in align.items() if k in PARAMS})
    x, y = x[0], y[0]
    canvas = Image.new('L', (w, h), 0); dr = ImageDraw.Draw(canvas)
    wid = int(round(align['s'] * (g['stroke'] + 2 * g['stray_dilate'])))
    for c in range(CELLS):
        for sg in range(7):
            if on[c, sg]:
                pts = list(zip(x[c, sg], y[c, sg]))
                # extend the sampled middle part to the full segment / 標本線を全長へ延ばす
                (xa, ya), (xb, yb) = pts[0], pts[-1]
                ext = g['seg_inset'] * align['s'] / max(1e-9, np.hypot(xb - xa, yb - ya))
                xa2, ya2 = xa - (xb - xa) * ext, ya - (yb - ya) * ext
                xb2, yb2 = xb + (xb - xa) * ext, yb + (yb - ya) * ext
                dr.line([(xa2, ya2), (xb2, yb2)], fill=255, width=wid)
    r = max(3, wid // 2); dr.ellipse([dp[0] - r, dp[1] - r, dp[0] + r, dp[1] + r], fill=255)
    pred = np.asarray(canvas) > 0
    lo, hi = np.rint(y.min() - 12).astype(int), np.rint(y.max() + 12).astype(int)
    xlo, xhi = np.rint(x.min() - 12).astype(int), np.rint(x.max() + 12).astype(int)
    zone = np.zeros((h, w), bool); zone[max(lo, 0):hi, max(xlo, 0):xhi] = True
    ink = (norm > (cal['thresholds']['stray_ink'] if ink_level is None else ink_level)) & zone
    return float((ink & ~pred).sum() / max(1, ink.sum())), int(ink.sum())


def read_frame(gray, cal):
    # one frame -> (value or None, ok, detail, debug) / 1 枚 -> （値、ok、理由、デバッグ情報）
    dark = darkness_map(gray, cal)
    block, why = find_block(dark, cal)
    if block is None:
        return None, 0, why, None
    norm = dark / block['level']
    align = fit(norm, block, cal)
    pname = align['start']
    digits, lev, on, reasons, dp = decode(norm, align, cal)
    stray, _ = stray_fraction(norm, align, on, dp, cal)
    if stray > cal['thresholds']['stray_max']:
        reasons.append('stray ink %.2f' % stray)
    faint, _ = stray_fraction(norm, align, on, dp, cal, cal['thresholds']['faint_ink'])
    if faint > cal['thresholds']['faint_max']:
        reasons.append('faint ink %.2f' % faint)
    if align['cost'] > cal['thresholds']['cost_max']:
        reasons.append('poor fit %.2f' % align['cost'])
    text = ''.join(digits)
    dbg = dict(align=align, digits=text, lev=lev, stray=stray, faint=faint, profile=pname, cal=cal)
    if reasons:
        return None, 0, '; '.join(reasons) + ' [%s]' % text.replace(' ', '_'), dbg
    value = float('%s%s.%s%s' % (digits[0].strip(), digits[1], digits[2], digits[3]))
    return (-value if minus_on(gray, cal) else value), 1, '', dbg


def minus_on(gray, cal):
    # minus sign: a bar at the far left of the LCD (cal['minus']: bar box and background boxes, frame pixels).
    # darkness = background median - 20th percentile in the bar box; about 90 with the sign, 9 without (2026-11-02b)
    # マイナス記号: 表示の左端の横棒。暗さ = 背景の中央値 - 棒の範囲の 20 パーセンタイル。記号ありで約 90、なしで約 9
    m = cal.get('minus')
    if not m:
        return False
    x0, x1, y0, y1 = m['bar']
    bg = np.concatenate([gray[b[2]:b[3], b[0]:b[1]].ravel() for b in m['background']])
    return float(np.median(bg) - np.percentile(gray[y0:y1, x0:x1], 20)) > m['darkness_min']


def frames_of(path, cal):
    # split a photo into its single frames / 写真を 1 枚ずつのフレームに分ける
    im = Image.open(path).convert('L')
    w, h = im.size; fw, fh = cal['frame_size']
    nx, ny = max(1, round(w / fw)), max(1, round(h / fh))
    for k in range(nx * ny):
        r, c = divmod(k, nx)
        yield (k if nx * ny > 1 else ''), np.asarray(im.crop((c * fw, r * fh, (c + 1) * fw, (r + 1) * fh)), float)


def run_dirs(top):
    # a run folder holds img/; otherwise look one level down / img/ を持つ階層を run とする
    if (top / 'img').is_dir():
        return [top]
    return sorted(p for p in top.iterdir() if (p / 'img').is_dir())


def process_run(run, cal):
    times = {}
    fcsv = run / 'frames.csv'
    if fcsv.exists():
        with open(fcsv) as f:
            for row in csv.DictReader(f):
                times[row['file']] = row['wall_time_s']
    out, n_ok = [], 0
    files = sorted((run / 'img').glob('*.jpg'))
    for p in files:
        rel = 'img/' + p.name
        for k, gray in frames_of(p, cal):
            value, ok, detail, _ = read_frame(gray, cal)
            n_ok += ok
            out.append([rel, k, times.get(rel, ''), '' if value is None else '%.2f' % value, ok, detail])
    with open(run / 'scale_readings.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['file', 'frame_in_tile', 'wall_time_s', 'value_g', 'ok', 'detail'])
        w.writerows(out)
    print('%-40s frames %5d  ok %5d  unreadable %5d' % (run, len(out), n_ok, len(out) - n_ok))
    return len(out), n_ok


def validate(csv_path, cal):
    # compare the reader with eye readings: every frame marked ok must match exactly
    # 目視の読みと照合する。ok とした枚は完全一致が必要
    root = pathlib.Path(csv_path).resolve().parent.parent
    cnt = dict(ok_match=0, ok_mismatch=0, ok_on_eye_unreadable=0, rejected_readable=0, rejected_unreadable=0)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            k = row['frame_in_tile']
            gray = next(g for kk, g in frames_of(root / row['file'], cal) if str(kk) == k)
            value, ok, detail, _ = read_frame(gray, cal)
            eye = row['value_by_eye']
            if ok and not eye:
                cnt['ok_on_eye_unreadable'] += 1; print('OK but unreadable by eye:', row['file'], k, value)
            elif ok and '%.2f' % value == eye:
                cnt['ok_match'] += 1
            elif ok:
                cnt['ok_mismatch'] += 1; print('MISMATCH:', row['file'], k, 'eye', eye, 'read', value)
            elif eye:
                cnt['rejected_readable'] += 1; print('rejected but readable by eye:', row['file'], k, eye, detail)
            else:
                cnt['rejected_unreadable'] += 1
    print(cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder', nargs='?'); ap.add_argument('--calib', default=str(CALIB_DEFAULT))
    ap.add_argument('--validate', metavar='CSV', help='check against eye readings (file, frame_in_tile, value_by_eye)')
    a = ap.parse_args()
    cal = json.load(open(a.calib))
    if a.validate:
        return validate(a.validate, cal)
    if not a.folder:
        sys.exit('give a run folder or --validate CSV')
    runs = run_dirs(pathlib.Path(a.folder))
    if not runs:
        sys.exit('no img/ folder under %s' % a.folder)
    for r in runs:
        process_run(r, cal)


if __name__ == '__main__':
    main()
