#!/usr/bin/env python3
"""Propeller thrust per steady step from scale photos (2026-11-02 motor bench).
2026-11-02 モータベンチ: 秤の写真読み取り値から各定常ステップのプロペラ推力を求める。

Pipeline / 手順:
  1. camera delay L from scale transitions (start / stop / duty change)
     秤の変化点（起動・停止・デューティ変更）からカメラ遅れ L を推定
  2. scale zero per motor-stopped period, linear interpolation in scene time
     停止区間ごとの秤ゼロ点を求め、シーン時刻で線形補間
  3. thrust per step in [start+1.0, end-0.3] (scene time), join with omega/VDD
     ステップ窓の推力を求め、omega・VDD と結合
  4. questions A (F vs omega), B (F vs VDD at fixed duty), C (hover duty vs VDD)
     設問 A（F-ω 関係）・B（固定デューティでの F-VDD）・C（ホバーデューティ-VDD）

Usage: python3 tools/analyze_thrust.py   (run from the repo root)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
G = 9.80665                      # g -> mN factor / g から mN への係数
WIN_START, WIN_END = 1.0, 0.3    # step window margins [s] / 窓の前後余白
ZERO_MARGIN = 1.5                # skip after stop [s] / 停止後に除く時間
L_FALLBACK = 4.3                 # only used if estimation fails / 推定失敗時のみ

BAT2 = [DATA / "2026-11-02_bat2" / n for n in ("A1", "B1", "B2", "B3", "C1", "C2", "C3")]
BAT3 = [DATA / "2026-11-02_bat3" / n for n in ("A4M", "B4M", "C4M")]
E2 = [DATA / "2026-11-02_E2" / "p"]
ALL_RUNS = BAT2 + BAT3 + E2


# ---------------------------------------------------------------- loading
def load_run(rd):
    """Load scale readings (ok only), events and step table.
    秤読み値(ok=1のみ)・イベント・ステップ表を読む。"""
    s = pd.read_csv(rd / "scale_readings.csv")
    s = s[s.ok == 1][["wall_time_s", "value_g"]].sort_values("wall_time_s")
    ev = pd.read_csv(rd / "events.csv")
    return s.reset_index(drop=True), ev, len(pd.read_csv(rd / "scale_readings.csv"))


def parse_commands(ev):
    """Return (starts, stops) from cmd rows. starts: time, duty, lap, step.
    cmd 行から開始・停止を取り出す。"""
    starts, stops = [], []
    for _, r in ev[ev.kind == "cmd"].iterrows():
        d = str(r.detail)
        m = re.match(r"motor (?:test \d+|all) (\d+) ", d)
        if m:
            nm = re.search(r"lap(\d+) step(\d+)", str(r.note))
            starts.append(dict(t=r.wall_time_s, duty=int(m.group(1)) / 100.0,
                               lap=int(nm.group(1)), step=int(nm.group(2))))
        elif d.startswith("motor stop"):
            stops.append(r.wall_time_s)
    return starts, stops


def step_table(starts, stops):
    """Step end = next command (start or stop). / ステップ終了 = 次のコマンド時刻。"""
    cmds = sorted([s["t"] for s in starts] + stops)
    out = []
    for s in starts:
        nxt = [c for c in cmds if c > s["t"] + 1e-6]
        out.append(dict(s, t_end=nxt[0] if nxt else np.nan))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- delay L
def crossing_delay(s, te, t_next, kind):
    """Delay of half-level crossing after command time te (arrival time based).
    コマンド時刻 te からの半値交差遅れ（到着時刻基準）。"""
    ta, v = s.wall_time_s.values, s.value_g.values
    pre_m = (ta >= te + 0.2) & (ta <= te + 3.3)
    hi = min(t_next + 3.8, te + 10.0) if np.isfinite(t_next) else te + 10.0
    post_m = (ta >= te + 6.0) & (ta <= hi)
    if pre_m.sum() < 2 or post_m.sum() < 2:
        return None
    pre, post = np.median(v[pre_m]), np.median(v[post_m])
    if abs(post - pre) < 0.8:
        return None
    mid = 0.5 * (pre + post)
    sgn = np.sign(post - pre)
    m = (ta >= te + 2.0) & (ta <= te + 8.5)
    tt, vv = ta[m], v[m]
    for i in range(1, len(tt)):
        if (vv[i] - mid) * sgn >= 0 and (vv[i - 1] - mid) * sgn < 0:
            f = (mid - vv[i - 1]) / (vv[i] - vv[i - 1])
            return tt[i - 1] + f * (tt[i] - tt[i - 1]) - te
    return None


def estimate_delay(rd, s, ev):
    starts, stops = parse_commands(ev)
    cmds = sorted([x["t"] for x in starts] + stops)
    stop_set = set(stops)
    rows = []
    for i, t in enumerate(cmds):
        t_next = cmds[i + 1] if i + 1 < len(cmds) else np.nan
        prev_stopped = i == 0 or cmds[i - 1] in stop_set
        if t in stop_set:
            kind = "stop"
        elif prev_stopped:
            kind = "start"
        else:
            kind = "change"
        d = crossing_delay(s, t, t_next, kind)
        if d is not None:
            rows.append(dict(run=rd.name, t=t, kind=kind, delay=d))
    return rows


# ---------------------------------------------------------------- zero
def zero_periods(s, ev, starts, stops, L):
    """Zero value per motor-stopped period (scene time). / 停止区間ごとのゼロ点。"""
    scene = s.wall_time_s.values - L
    v = s.value_g.values
    first_cmd = min([x["t"] for x in starts] + stops)
    t_end_all = scene.max() + 1.0
    periods = [(scene.min() + 0.5, first_cmd)]            # pre-roll
    ordered = sorted([("start", x["t"]) for x in starts] + [("stop", t) for t in stops],
                     key=lambda x: x[1])
    running, t_stop = False, None
    for kind, t in ordered:
        if kind == "start":
            if not running and t_stop is not None:
                periods.append((t_stop + ZERO_MARGIN, t))
            running, t_stop = True, None
        else:
            if running:
                running, t_stop = False, t
    if not running and t_stop is not None:
        periods.append((t_stop + ZERO_MARGIN, t_end_all))
    out = []
    for a, b in periods:
        m = (scene >= a) & (scene <= b)
        if m.sum() >= 2:
            out.append(dict(t=float(scene[m].mean()), zero=float(np.median(v[m])),
                            sd=float(np.std(v[m])), n=int(m.sum()), a=a, b=b))
    return out


# ---------------------------------------------------------------- per run
def read_electrical(rd, steps):
    """Join omega, VDD from analysis_steps.csv (or points.csv for E2).
    omega・VDD を analysis_steps.csv（E2 は points.csv）から結合。"""
    f = rd / "analysis_steps.csv"
    if f.exists():
        a = pd.read_csv(f)
        a = a[["lap", "step", "duty_cmd", "n_motors", "VDD", "omega_new"]].rename(
            columns={"omega_new": "omega"})
    else:
        p = pd.read_csv(rd / "points.csv")
        p = p[p.skipped == 0] if "skipped" in p else p
        a = pd.DataFrame(dict(lap=p.lap, step=p.step, duty_cmd=p.duty_cmd / 100.0,
                              n_motors=1, VDD=p.VDD_V, omega=p.omega_rad_s))
        # wrong FFT line in E2/p lap1 duty 20 -> ignore / 誤った FFT 線を除外
        a.loc[(a.lap == 1) & (a.step == 1), "omega"] = np.nan
    return steps.merge(a[["lap", "step", "n_motors", "VDD", "omega"]], on=["lap", "step"], how="left")


def process_run(rd, L):
    s, ev, n_total = load_run(rd)
    starts, stops = parse_commands(ev)
    steps = step_table(starts, stops)
    zp = zero_periods(s, ev, starts, stops, L)
    zt = np.array([z["t"] for z in zp])
    zv = np.array([z["zero"] for z in zp])
    scene = s.wall_time_s.values - L
    v = s.value_g.values
    rows = []
    for _, st in steps.iterrows():
        a, b = st.t + WIN_START, st.t_end - WIN_END
        m = (scene >= a) & (scene <= b)
        n = int(m.sum())
        rec = dict(run=rd.name, lap=st.lap, step=st.step, duty=st.duty, t_start=st.t,
                   t_end=st.t_end, n_frames=n)
        if n >= 1:
            z = np.interp(scene[m], zt, zv)
            th = v[m] - z
            rec.update(zero_g=float(z.mean()), thrust_g=float(th.mean()),
                       thrust_sd_g=float(th.std(ddof=1)) if n > 1 else np.nan)
            if n >= 3 and np.ptp(scene[m]) > 0.5:
                rec["slope_g_per_s"] = float(np.polyfit(scene[m], th, 1)[0])
            h = n // 2
            rec["half_diff_g"] = float(th[h:].mean() - th[:h].mean()) if n >= 4 else np.nan
            rec["thrust_mN"] = rec["thrust_g"] * G
        rows.append(rec)
    df = read_electrical(rd, pd.DataFrame(rows))
    return df, zp, n_total, len(s)


# ---------------------------------------------------------------- stats helpers
def ols(x, y):
    """OLS slope with standard error. / 傾きとその標準誤差。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3 or np.ptp(x) == 0:
        return np.nan, np.nan, n
    A = np.vstack([x, np.ones(n)]).T
    c, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    s2 = (r @ r) / (n - 2)
    se = np.sqrt(s2 / ((x - x.mean()) ** 2).sum())
    return c[0], se, n


def fit_f_omega(w, F):
    """F = c2 w^2 + c3 w^3 minimizing relative error. / 相対誤差最小の最小二乗。"""
    A = np.vstack([w ** 2, w ** 3]).T / F[:, None]
    c, *_ = np.linalg.lstsq(A, np.ones(len(F)), rcond=None)
    return c


def group_of(run):
    return "warm" if run.startswith("B") else "cold"


# ---------------------------------------------------------------- main
def main():
    out_md = []
    P = out_md.append

    # 1. delay (events pooled over all runs; scene time needs L before zero)
    delay_rows = []
    for rd in ALL_RUNS:
        if not (rd / "scale_readings.csv").exists():
            continue
        s, ev, _ = load_run(rd)
        delay_rows += estimate_delay(rd, s, ev)
    dd = pd.DataFrame(delay_rows)
    L = float(dd.delay.median()) if len(dd) else L_FALLBACK
    P("# Thrust analysis 2026-11-02 (generated by tools/analyze_thrust.py)\n")
    P("## 1. Camera delay L (arrival time - scene time)\n")
    P(f"All events pooled (half-level crossing after command time): n={len(dd)}, "
      f"median **L = {L:.2f} s**, std {dd.delay.std():.2f} s, "
      f"IQR [{dd.delay.quantile(.25):.2f}, {dd.delay.quantile(.75):.2f}] s\n")
    P("| kind | n | median [s] | std [s] |\n|---|---|---|---|")
    for k, g in dd.groupby("kind"):
        P(f"| {k} | {len(g)} | {g.delay.median():.2f} | {g.delay.std():.2f} |")
    # L differs between runs (camera buffering), so use the per-run median
    # カメラ遅れは run ごとに違うため、run ごとの中央値を使う
    L_run = {r: float(g.delay.median()) if len(g) >= 2 else L for r, g in dd.groupby("run")}
    P("\nPer run (scene time uses the per-run median): " + ", ".join(
        f"{r}={g.delay.median():.2f}+-{g.delay.std():.2f}(n={len(g)})"
        for r, g in dd.groupby("run")) + "\n")
    print(f"L = {L:.3f} s  (n={len(dd)}, std={dd.delay.std():.3f})", L_run)

    # 2/3. per run
    P("## 2. Scale zero per run\n")
    P("| run | zero periods | zero values [g] | drift last-first [g] | range [g] | rejected frames (ok=0) |\n|---|---|---|---|---|---|")
    allsteps = []
    for rd in ALL_RUNS:
        if not (rd / "scale_readings.csv").exists():
            continue
        df, zp, n_total, n_ok = process_run(rd, L_run.get(rd.name, L))
        df.to_csv(rd / "thrust_steps.csv", index=False)
        allsteps.append(df)
        zs = [z["zero"] for z in zp]
        P(f"| {rd.parent.name}/{rd.name} | {len(zp)} | "
          + " ".join(f"{z:.2f}" for z in zs[:8]) + (" ..." if len(zs) > 8 else "")
          + f" | {zs[-1] - zs[0]:+.2f} | {max(zs) - min(zs):.2f} | {n_total - n_ok}/{n_total} |")
    steps = pd.concat(allsteps, ignore_index=True)
    steps["cond"] = steps.run.map(group_of)
    steps["set"] = steps.run.map(lambda r: "bat3" if r.endswith("M") else ("E2" if r == "p" else "bat2"))

    # suspicious steps / 怪しいステップ
    P("\n## 3. Step quality\n")
    bad = steps[(steps.n_frames < 4) | (steps.slope_g_per_s.abs() > 0.25) | (steps.half_diff_g.abs() > 0.3)]
    P(f"Steps total {len(steps)}; flagged (n_frames<4 or |slope|>0.25 g/s or |half diff|>0.3 g): {len(bad)}\n")
    P("| run | lap | step | duty | n | thrust [g] | slope [g/s] | half diff [g] |\n|---|---|---|---|---|---|---|---|")
    for _, r in bad.iterrows():
        P(f"| {r.run} | {r.lap} | {r.step} | {r.duty:.2f} | {r.n_frames} | {r.thrust_g:.2f} | "
          f"{r.slope_g_per_s:+.2f} | {r.half_diff_g:+.2f} |")
    P(f"\nFull per-step tables: `<run>/thrust_steps.csv`. Typical step: n_frames median "
      f"{steps.n_frames.median():.0f}, sd median {steps.thrust_sd_g.median():.3f} g.\n")

    # A. F vs omega (bat2 single motor)
    b2 = steps[(steps.set == "bat2") & steps.omega.notna() & steps.thrust_g.notna()].copy()
    w, F = b2.omega.values, b2.thrust_g.values
    c = fit_f_omega(w, F)
    b2["fit"] = c[0] * w ** 2 + c[1] * w ** 3
    b2["res_pct"] = 100 * (F - b2.fit) / b2.fit
    P("## 4. Question A: thrust vs omega (bat2 single-motor runs)\n")
    P(f"Fit F[g] = c2 w^2 + c3 w^3 (relative LSQ), n={len(b2)}: c2={c[0]:.4e}, c3={c[1]:.4e} "
      f"(w in rad/s)\n")
    P(f"Residual %: mean {b2.res_pct.mean():+.2f}, std {b2.res_pct.std():.2f}, "
      f"max|.| {b2.res_pct.abs().max():.2f}; fraction within +-2 %: "
      f"{(b2.res_pct.abs() <= 2).mean() * 100:.0f} %\n")
    P("| subset | n | mean res % | std % | max abs % |\n|---|---|---|---|---|")
    for name, g in list(b2.groupby("cond")) + list(b2.groupby("run")):
        P(f"| {name} | {len(g)} | {g.res_pct.mean():+.2f} | {g.res_pct.std():.2f} | {g.res_pct.abs().max():.2f} |")
    sl, se, n = ols(b2.VDD, b2.res_pct)
    P(f"\nResidual % vs VDD (all): slope {sl:+.2f} +- {se:.2f} %/V (n={n})")
    # with warm/cold dummy and omega-bin control / 温冷ダミー付き
    X = np.vstack([b2.VDD, (b2.cond == "warm").astype(float), np.ones(len(b2))]).T
    cc, *_ = np.linalg.lstsq(X, b2.res_pct.values, rcond=None)
    r = b2.res_pct.values - X @ cc
    cov = (r @ r) / (len(b2) - 3) * np.linalg.inv(X.T @ X)
    P(f"Residual % = a*VDD + b*warm + c: a={cc[0]:+.2f}+-{np.sqrt(cov[0, 0]):.2f} %/V, "
      f"b(warm)={cc[1]:+.2f}+-{np.sqrt(cov[1, 1]):.2f} %\n")
    P("Residual % by duty and condition (mean):\n")
    piv = b2.pivot_table(index="duty", columns="cond", values="res_pct", aggfunc="mean").round(2)
    P(piv.to_markdown() + "\n")
    P("Residual % by lap for B runs (mean over B1-B3):\n")
    P(b2[b2.cond == "warm"].pivot_table(index="duty", columns="lap", values="res_pct", aggfunc="mean").round(2).to_markdown() + "\n")
    # diagnostic: common coefficient a, free offset per (run, lap): F = a w^2 - b_k
    # 診断: 共通係数 a、(run, lap) ごとに自由なオフセット b_k
    keys = sorted(set(zip(b2.run, b2.lap)))
    X = np.zeros((len(b2), 1 + len(keys)))
    X[:, 0] = b2.omega.values ** 2
    for i, k in enumerate(keys):
        X[:, 1 + i] = -((b2.run == k[0]) & (b2.lap == k[1])).values.astype(float)
    cf, *_ = np.linalg.lstsq(X, b2.thrust_g.values, rcond=None)
    rr = b2.thrust_g.values - X @ cf
    P("Diagnostic F = a*w^2 - b_k with a free offset b_k per (run, lap), common a: "
      f"a={cf[0]:.4e} g/(rad/s)^2, residual rms {np.sqrt((rr ** 2).mean()):.3f} g "
      f"({100 * np.sqrt(((rr / b2.thrust_g.values) ** 2).mean()):.1f} % rms of F).\n")
    P("| run | lap | b_k [g] | mean VDD [V] | mean zero [g] |\n|---|---|---|---|---|")
    for i, k in enumerate(keys):
        g = b2[(b2.run == k[0]) & (b2.lap == k[1])]
        P(f"| {k[0]} | {k[1]} | {cf[1 + i]:+.2f} | {g.VDD.mean():.3f} | {g.zero_g.mean():.2f} |")
    P("")
    # per-run log-log exponent d ln F / d ln w / run ごとの対数傾き
    P("Per (run, lap) local exponent n = d lnF/d ln w (F ~ w^n; 2 expected without offset): "
      + ", ".join(f"{r}L{l}:{np.polyfit(np.log(g.omega), np.log(g.thrust_g), 1)[0]:.1f}"
                  for (r, l), g in b2.groupby(["run", "lap"]) if (g.thrust_g > 0.5).all()) + "\n")
    bk = pd.DataFrame(dict(b=cf[1:], vdd=[b2[(b2.run == k[0]) & (b2.lap == k[1])].VDD.mean() for k in keys],
                           zero=[b2[(b2.run == k[0]) & (b2.lap == k[1])].zero_g.mean() for k in keys]))
    Xb = np.vstack([bk.zero, bk.vdd, np.ones(len(bk))]).T
    cb, *_ = np.linalg.lstsq(Xb, bk.b.values, rcond=None)
    rb = bk.b.values - Xb @ cb
    P(f"Offset b_k regressed on [zero, VDD]: b = {cb[0]:+.2f}*zero {cb[1]:+.2f}*VDD {cb[2]:+.2f} "
      f"(rms resid {np.sqrt((rb ** 2).mean()):.2f} g); corr(b, zero)={np.corrcoef(bk.b, bk.zero)[0, 1]:.3f}, "
      f"corr(b, VDD)={np.corrcoef(bk.b, bk.vdd)[0, 1]:.3f}\n")
    # E2 (low duties, stop after every step) overlay / E2 を同じ曲線に重ねる
    e2 = steps[(steps.set == "E2") & steps.omega.notna() & steps.thrust_g.notna()]
    if len(e2):
        fe = c[0] * e2.omega ** 2 + c[1] * e2.omega ** 3
        P("E2/p (duties 0.20-0.60, extrapolation check against the bat2 curve): "
          + ", ".join(f"d{d:.2f}:{x:+.1f}%" for d, x in
                      zip(e2.duty, 100 * (e2.thrust_g - fe) / fe)) + "\n")

    # B. thrust vs VDD at fixed duty
    P("## 5. Question B: thrust and omega vs VDD at fixed duty (bat2)\n")
    P("Slopes in % per volt (relative to group mean) +- standard error; "
      "ratio = thrust slope / omega slope (2 if F ~ w^2).\n")
    P("| duty | group | n | VDD range [V] | thrust %/V | omega %/V | ratio |\n|---|---|---|---|---|---|---|")
    for d, gd in b2.groupby("duty"):
        for gname, g in (("cold", gd[gd.cond == "cold"]), ("warm", gd[gd.cond == "warm"]), ("all", gd)):
            if len(g) < 3:
                continue
            sF, eF, n = ols(g.VDD, 100 * g.thrust_g / g.thrust_g.mean())
            sw, ew, _ = ols(g.VDD, 100 * g.omega / g.omega.mean())
            P(f"| {d:.2f} | {gname} | {n} | {g.VDD.min():.3f}-{g.VDD.max():.3f} | "
              f"{sF:+.1f} +- {eF:.1f} | {sw:+.1f} +- {ew:.1f} | {sF / sw if sw else np.nan:.2f} |")
    P("")
    # within-run: B laps at the same duty, run-demeaned (removes run-to-run offsets)
    # run 内比較: B の lap 間、同一デューティ、run 内で平均除去
    xs, yF, yw = [], [], []
    for (r, d), gd in b2[b2.cond == "warm"].groupby(["run", "duty"]):
        xs += list(gd.VDD - gd.VDD.mean())
        yF += list(100 * (gd.thrust_g - gd.thrust_g.mean()) / gd.thrust_g.mean())
        yw += list(100 * (gd.omega - gd.omega.mean()) / gd.omega.mean())
    sF, eF, n = ols(xs, yF)
    sw, ew, _ = ols(xs, yw)
    P(f"Within-run (B1-B3, laps 1-4 at the same duty, run-demeaned): thrust {sF:+.1f} +- {eF:.1f} %/V, "
      f"omega {sw:+.1f} +- {ew:.1f} %/V (n={n}), ratio {sF / sw:.2f}\n")
    P("Pooled over duties (each duty normalised to its group mean, all bat2):\n")
    for gname in ("cold", "warm", "all"):
        g = b2 if gname == "all" else b2[b2.cond == gname]
        xs, yF, yw = [], [], []
        for d, gd in g.groupby("duty"):
            xs += list(gd.VDD - gd.VDD.mean())
            yF += list(100 * gd.thrust_g / gd.thrust_g.mean() - 100)
            yw += list(100 * gd.omega / gd.omega.mean() - 100)
        sF, eF, n = ols(xs, yF)
        sw, ew, _ = ols(xs, yw)
        P(f"- {gname}: thrust {sF:+.1f} +- {eF:.1f} %/V, omega {sw:+.1f} +- {ew:.1f} %/V (n={n}), ratio {sF / sw:.2f}")
    P("")

    # C. four motors
    P("## 6. Question C: four motors, hover duty\n")
    b3 = steps[(steps.set == "bat3") & steps.thrust_g.notna()]
    hov_rows = []
    P("Total thrust [g] vs duty per lap:\n")
    for (run, lap), g in b3.groupby(["run", "lap"]):
        g = g.sort_values("duty")
        P(f"- {run} lap{lap}: " + ", ".join(f"{d:.2f}:{t:.1f}" for d, t in zip(g.duty, g.thrust_g)))
        for W in (37.0, 36.4):
            d_, t_, v_ = g.duty.values, g.thrust_g.values, g.VDD.values
            hit = None
            for i in range(len(d_) - 1):
                if (t_[i] - W) * (t_[i + 1] - W) <= 0 and t_[i] != t_[i + 1]:
                    f = (W - t_[i]) / (t_[i + 1] - t_[i])
                    hit = (d_[i] + f * (d_[i + 1] - d_[i]), v_[i] + f * (v_[i + 1] - v_[i]))
                    break
            hov_rows.append(dict(run=run, lap=lap, W=W, duty=hit[0] if hit else np.nan,
                                 VDD=hit[1] if hit else np.nan, tmax=t_.max()))
    hv = pd.DataFrame(hov_rows)
    P("\n| run | lap | weight [g] | hover duty | VDD at hover [V] |\n|---|---|---|---|---|")
    for _, r in hv.iterrows():
        res = f"{r.duty:.4f} | {r.VDD:.3f}" if r.duty == r.duty else f"not bracketed (max {r.tmax:.1f} g) | -"
        P(f"| {r.run} | {r.lap:.0f} | {r.W:.1f} | {res} |")
    P("")
    for W in (37.0, 36.4):
        h = hv[(hv.W == W) & hv.duty.notna()]
        sl, se, n = ols(h.VDD, h.duty)
        P(f"- W={W}: duty vs VDD slope {sl:+.4f} +- {se:.4f} duty/V (n={n}); "
          f"duty range {h.duty.min():.3f}-{h.duty.max():.3f}, VDD {h.VDD.min():.3f}-{h.VDD.max():.3f}")
    P("")
    P("Optional inputs not processed: bat1 exp_auto (pwm_c02_d*); E2/p has no scale_readings.csv "
      "(reader not run on its frames), so it is skipped (the loader supports it once the file exists).\n")

    (DATA / "2026-11-02_summary_thrust.md").write_text("\n".join(out_md))
    print("\n".join(out_md))


if __name__ == "__main__":
    sys.exit(main())
