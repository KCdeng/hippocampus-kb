"""kb_soak_test.py — 海马体 8310 的长期并发 + 热重载压测（只发查询，不动任何索引）。

为什么手动压：个人服务器的真实负载是"几个智能体同时问"，单次冒烟测不出线程堆积、
内存漂移和热重载期间的抖动。这里把并发、时长、判定阈值都写死，跑完给
P50/P95/P99、错误数、内存/线程/日志增量，而不是"看起来没事"。

判据在跑之前就声明（见 THRESHOLDS）：errors==0 是硬线（FAIL 就退出码 1）；
P95、内存/线程增量超线只报 WARN——这台机器只有 CPU ONNX，几十分钟的样本证明不了
"没有泄漏"，只能证明"这段时间没在涨"。

查询一律带 replay:true：不写 query_log、不计命中，压测不污染真实统计——脚本末尾
会核对 query_log 确实没长，顺带把这个开关本身也验一遍。

用法:
  venv/Scripts/python.exe -X utf8 kb/kb_soak_test.py --minutes-steady 18 --minutes-burst 3
  ... --skip-reload --json kb/scratch/soak.json
"""
import argparse, json, os, random, sys, threading, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from kb_health_probe import listeners, proc_info  # noqa: E402

THRESHOLDS = {"max_errors": 0, "steady_p95_ms": 2500, "burst_p95_ms": 8000,
              "mem_growth_mb_warn": 300, "threads_growth_warn": 20}
SAMPLE_EVERY = 30

PROMPTS = {"main": os.path.join(HERE, "eval_set.json"),
           "zh": os.path.join(BASE, "kb_zh", "eval_set.json"),
           "comp": os.path.join(BASE, "kb_comp", "eval_set.json")}
# 负载构成：主库（带 mem/epi 尾部）占大头，竞品/专利次之——和实际派活比例一致
MIX = [("main", 0.55), ("comp", 0.25), ("zh", 0.20)]


def load_pool():
    return {kb: [e["question"] for e in json.load(open(p, encoding="utf-8"))]
            for kb, p in PROMPTS.items()}


def pick(mix):
    r, acc = random.random(), 0.0
    for kb, w in mix:
        acc += w
        if r <= acc:
            return kb
    return mix[0][0]


def search(port, kb, q, k=8, timeout=120):
    body = {"query": q, "k": k, "replay": True}
    if kb != "main":
        body["kb"] = kb
    req = urllib.request.Request(f"http://127.0.0.1:{port}/search",
                                 json.dumps(body).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        ms = (time.time() - t0) * 1000
        if isinstance(data, dict):
            return False, ms, f"svc_error:{data.get('error')}"
        return True, ms, f"{kb}:hits={len(data)}"
    except (urllib.error.URLError, OSError, ValueError) as e:
        return False, (time.time() - t0) * 1000, f"{kb}:{type(e).__name__}:{str(e)[:80]}"


def post(port, path, body, timeout=300):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 json.dumps(body).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        return 0, {"error": f"{type(e).__name__}:{str(e)[:90]}"}


def pct(xs, p):
    if not xs:
        return None
    ys = sorted(xs)
    return ys[min(int(round(p / 100.0 * (len(ys) - 1))), len(ys) - 1)]


class Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.rows = []

    def add(self, phase, ok, ms, note, t):
        with self.lock:
            self.rows.append({"phase": phase, "ok": ok, "ms": ms, "note": note, "t": t})

    def slice(self, phase):
        with self.lock:
            return [r for r in self.rows if r["phase"] == phase]


def worker(port, pool, phase, deadline, think, rec, stop):
    while not stop.is_set() and time.time() < deadline:
        kb = pick(MIX)
        ok, ms, note = search(port, kb, random.choice(pool[kb]))
        rec.add(phase, ok, ms, note, time.time())
        if think:
            time.sleep(random.uniform(think * 0.5, think * 1.5))


def get_json(port, path, timeout=20):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def sample_loop(port, pid, start, deadline, stop, samples):
    """定时抓服务进程内存/线程，看增长曲线；采样本身失败也要留痕。"""
    while not stop.is_set() and time.time() < deadline:
        try:
            info = proc_info(pid) or {}
            stats = get_json(port, "/stats")
            samples.append({"t": round(time.time() - start, 1),
                            "priv_mb": round(info.get("pm", 0) / 1048576, 1),
                            "threads": info.get("th"),
                            "server_n": (stats or {}).get("perf", {}).get("n")})
        except Exception as e:                                  # noqa: BLE001
            samples.append({"t": round(time.time() - start, 1), "sample_error": str(e)[:80]})
        time.sleep(SAMPLE_EVERY)


def size_of(name):
    p = os.path.join(HERE, name)
    return os.path.getsize(p) if os.path.exists(p) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8310)
    ap.add_argument("--minutes-steady", type=float, default=18.0)
    ap.add_argument("--minutes-burst", type=float, default=3.0)
    ap.add_argument("--steady-workers", type=int, default=4)
    ap.add_argument("--burst-workers", type=int, default=8)
    ap.add_argument("--think", type=float, default=1.5, help="稳态每路查询之间的间隔秒（模拟智能体节奏）")
    ap.add_argument("--reload-after", type=float, default=8.0, help="稳态开跑多少分钟后触发 /reload all")
    ap.add_argument("--skip-reload", action="store_true")
    ap.add_argument("--json", default=os.path.join(HERE, "scratch", "soak.json"))
    a = ap.parse_args()

    rows, lerr = listeners(a.port)
    if lerr:
        raise SystemExit(f"netstat 失败: {lerr}")
    if not rows:
        raise SystemExit(f"端口 {a.port} 无监听，先跑 _start.ps1")
    if len(rows) > 1:
        raise SystemExit(f"端口 {a.port} 有多个监听者 {[r[1] for r in rows]}，先查清再压")
    pid = int(rows[0][1])
    pool, rec, samples, stop = load_pool(), Recorder(), [], threading.Event()

    base = proc_info(pid) or {}
    qlog_start, errlog_start = size_of("query_log.jsonl"), size_of("serve.err.log")
    print(f"[soak] pid={pid} 起始 私有内存={round(base.get('pm', 0) / 1048576, 1)}MB "
          f"线程={base.get('th')} query_log={qlog_start}B")

    start = time.time()
    total_deadline = start + (a.minutes_steady + a.minutes_burst) * 60 + 90
    threading.Thread(target=sample_loop, args=(a.port, pid, start, total_deadline, stop, samples),
                     daemon=True).start()

    # 相位 1：稳态并发（若干路 + 思考间隔），中途打一发 /reload all
    reload_info = None
    dl = start + a.minutes_steady * 60
    ts = [threading.Thread(target=worker, args=(a.port, pool, "steady", dl, a.think, rec, stop),
                           daemon=True) for _ in range(a.steady_workers)]
    for t in ts:
        t.start()
    reload_at = start + a.reload_after * 60
    while time.time() < dl:
        if reload_info is None and not a.skip_reload and time.time() >= reload_at:
            t0 = time.time()
            code, body = post(a.port, "/reload", {"zone": "all"}, timeout=900)
            info = proc_info(pid) or {}
            reload_info = {"http": code, "body": body, "ms": round((time.time() - t0) * 1000),
                           "priv_mb_during": round(info.get("pm", 0) / 1048576, 1)}
            print(f"[soak] /reload all → {json.dumps(reload_info, ensure_ascii=False)}")
        time.sleep(1)
    for t in ts:
        t.join(timeout=20)

    # 相位 2：无间隔突发（8 路硬怼，看线程池与 ONNX 争抢的尾巴）
    dl2 = time.time() + a.minutes_burst * 60
    ts2 = [threading.Thread(target=worker, args=(a.port, pool, "burst", dl2, 0, rec, stop),
                            daemon=True) for _ in range(a.burst_workers)]
    for t in ts2:
        t.start()
    while time.time() < dl2:
        time.sleep(1)
    stop.set()
    for t in ts2:
        t.join(timeout=15)

    end = proc_info(pid) or {}
    report(rec, samples, base, end, a, reload_info, pid, qlog_start, errlog_start)


def report(rec, samples, base, end, a, reload_info, pid, qlog_start, errlog_start):
    out = {"pid": pid, "thresholds": THRESHOLDS, "args": {k: v for k, v in vars(a).items()},
           "phases": {}, "verdicts": [], "samples": samples, "errors_detail": [],
           "reload": reload_info}
    n_err = 0
    for phase, workers in (("steady", a.steady_workers), ("burst", a.burst_workers)):
        rows = rec.slice(phase)
        ok_ms = [r["ms"] for r in rows if r["ok"]]
        errs = [r for r in rows if not r["ok"]]
        n_err += len(errs)
        span = (max(r["t"] for r in rows) - min(r["t"] for r in rows)) if rows else 0
        out["phases"][phase] = {"workers": workers, "n": len(rows), "errors": len(errs),
                                "p50": pct(ok_ms, 50), "p95": pct(ok_ms, 95), "p99": pct(ok_ms, 99),
                                "max": max(ok_ms) if ok_ms else None,
                                "rps": round(len(rows) / span, 2) if span else None,
                                "span_s": round(span, 1)}
        out["errors_detail"].extend(errs[:25])

    def add(level, name, msg):
        out["verdicts"].append([level, name, msg])

    if n_err > THRESHOLDS["max_errors"]:
        add("FAIL", "errors", f"{n_err} 次请求失败（阈值 {THRESHOLDS['max_errors']}）")
    if out["phases"]["steady"]["n"] < 20:
        add("FAIL", "sample_size", f"稳态仅 {out['phases']['steady']['n']} 次请求，样本太少")
    s95, b95 = out["phases"]["steady"]["p95"], out["phases"]["burst"]["p95"]
    if s95 and s95 > THRESHOLDS["steady_p95_ms"]:
        add("WARN", "steady_p95", f"{round(s95)}ms > {THRESHOLDS['steady_p95_ms']}ms")
    if b95 and b95 > THRESHOLDS["burst_p95_ms"]:
        add("WARN", "burst_p95", f"{round(b95)}ms > {THRESHOLDS['burst_p95_ms']}ms")
    if reload_info and reload_info["http"] != 200:
        add("FAIL", "reload", f"POST /reload all → {reload_info['http']} {reload_info['body']}")

    d_priv = round((end.get("pm", 0) - base.get("pm", 0)) / 1048576, 1)
    d_th = (end.get("th") or 0) - (base.get("th") or 0)
    peak = max([s.get("priv_mb", 0) for s in samples], default=None)
    out["growth"] = {"start_priv_mb": round(base.get("pm", 0) / 1048576, 1),
                     "end_priv_mb": round(end.get("pm", 0) / 1048576, 1), "delta_priv_mb": d_priv,
                     "peak_priv_mb": peak,
                     "start_threads": base.get("th"), "end_threads": end.get("th"), "delta_threads": d_th}
    if d_priv > THRESHOLDS["mem_growth_mb_warn"]:
        add("WARN", "mem_growth", f"私有内存 +{d_priv}MB > {THRESHOLDS['mem_growth_mb_warn']}MB")
    if d_th > THRESHOLDS["threads_growth_warn"]:
        add("WARN", "threads_growth", f"线程 +{d_th} > {THRESHOLDS['threads_growth_warn']}")

    qlog_end, errlog_end = size_of("query_log.jsonl"), size_of("serve.err.log")
    out["logs"] = {"query_log_start": qlog_start, "query_log_end": qlog_end,
                   "serve_err_start": errlog_start, "serve_err_end": errlog_end}
    # replay:true 生效的话，query_log 一字节都不该长
    if qlog_end != qlog_start:
        add("WARN", "replay_flag", f"带 replay:true 的压测让 query_log 从 {qlog_start}B 长到 {qlog_end}B")

    n_fail = sum(1 for v in out["verdicts"] if v[0] == "FAIL")
    n_warn = sum(1 for v in out["verdicts"] if v[0] == "WARN")
    print(f"\n[soak] 总请求 {sum(out['phases'][p]['n'] for p in ('steady', 'burst'))} "
          f"失败 {n_err} → {'FAIL' if n_fail else 'PASS'}（{n_fail} FAIL / {n_warn} WARN）")
    for ph in ("steady", "burst"):
        p = out["phases"][ph]
        r = lambda x: x and round(x)
        print(f"  {ph:6} workers={p['workers']} n={p['n']} err={p['errors']} "
              f"P50={r(p['p50'])} P95={r(p['p95'])} P99={r(p['p99'])} max={r(p['max'])} "
              f"rps={p['rps']} span={p['span_s']}s")
    g = out["growth"]
    print(f"  内存 {g['start_priv_mb']}→{g['end_priv_mb']}MB（峰值 {g['peak_priv_mb']}，Δ{d_priv}）"
          f" 线程 {g['start_threads']}→{g['end_threads']}（Δ{d_th}）")
    for st, name, msg in out["verdicts"]:
        print(f"  [{st}] {name}: {msg}")
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    json.dump(out, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"json → {a.json}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
