"""kb_health_probe.py — 海马体 8310 的一次性健康探测（只读，不改任何生产件）。

为什么要它：/health 只回答"我起来了"，/stats 只比较"已入库文件"的新鲜度。
个人服务器上真正会踩的坑是别的：进程是不是还在跑旧代码、内存有没有飘、五区
溯源与块数对不对得上、正式文档写了但没入库（灯是绿的但查不到）、只增档的日志
有没有把轮转跑起来。这个探测把这些一次跑完并给退出码，便于人或计划任务判读。

用法:
  venv/Scripts/python.exe -X utf8 kb/kb_health_probe.py
  ... --port 8310 --json kb/scratch/health_probe.json --skip-coverage
退出码：0=没有 FAIL（可能有 WARN），1=有 FAIL。
"""
import argparse, datetime, hashlib, json, os, subprocess, sys, time
import urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)

EXPECT_MODEL = {"main": "bge-m3-onnx", "mem": "bge-m3-onnx", "epi": "bge-m3-onnx",
                "zh": "intfloat/multilingual-e5-large", "comp": "bge-m3-onnx"}
ZONE_FILES = {
    "main": (os.path.join(HERE, "chunks.jsonl"), os.path.join(HERE, "vec_m3.npy")),
    "mem": (os.path.join(HERE, "mem", "chunks.jsonl"), os.path.join(HERE, "mem", "vec.npy")),
    "epi": (os.path.join(HERE, "epi", "chunks.jsonl"), os.path.join(HERE, "epi", "vec.npy")),
    "zh": (os.path.join(BASE, "kb_zh", "chunks.jsonl"), os.path.join(BASE, "kb_zh", "vec.npy")),
    "comp": (os.path.join(BASE, "kb_comp", "chunks.jsonl"), os.path.join(BASE, "kb_comp", "vec.npy")),
}
SMOKE = [("main", "保压压力切换时机", 1), ("main", "注塑机 螺杆 熔胶 背压 控制", 1),
         ("zh", "全电动注塑机 专利", 1), ("comp", "同业厂商 伺服 注塑机", 1)]
PRIVATE_MB_WARN = 6000
THREADS_WARN = 200
LAT_WARN_MS = 2000


def get(port, path, body=None, timeout=25):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def rows_of(chunks_path):
    if not os.path.exists(chunks_path):
        return None
    with open(chunks_path, encoding="utf-8") as r:
        return sum(1 for l in r if l.strip())


def archives_of(path):
    """同名轮转/留档件（query_log.jsonl.1、serve.err.log.20261104-…）。
    只在同目录里找，按名字排序：`.1/.2/.3` 与时间戳两种命名都是时间序。"""
    d, name = os.path.split(path)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.startswith(name + "."))


def listeners(port):
    """返回该端口 LISTEN 的 (本地地址, pid) 列表。用 netstat 免依赖。"""
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"netstat 失败: {e}"
    rows = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "TCP" and parts[3].upper() == "LISTENING" \
                and parts[1].rsplit(":", 1)[-1] == str(port):
            rows.append((parts[1], parts[4]))
    return rows, None


def proc_info(pid):
    """私有内存/线程/启动时刻。PowerShell 拿，别用 workingset——换页会把内存报小。"""
    ps = ("$p=Get-Process -Id %s; [pscustomobject]@{pm=$p.PrivateMemorySize64;th=$p.Threads.Count;"
          "st=$p.StartTime.ToString('yyyy-MM-dd HH:mm:ss')} | ConvertTo-Json -Compress" % int(pid))
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=30).stdout.strip()
        return json.loads(out) if out else None
    except Exception as e:                                    # noqa: BLE001
        return None


def check(name, status, detail):
    return {"check": name, "status": status, "detail": detail}


def disk_budget(path, max_bytes, keep):
    """一份"只增档"的轮转体检：当前档不超上限、留档份数不超 keep。
    上限的单一来源是 kb_serve 的常量（同一组环境变量），探测不另抄一份数字。"""
    cur = os.path.getsize(path) if os.path.exists(path) else 0
    arch = archives_of(path)
    tot = cur + sum(os.path.getsize(os.path.join(os.path.dirname(path), f)) for f in arch)
    problems = []
    if cur > max_bytes:
        problems.append(f"当前档 {cur} 字节 > 上限 {max_bytes}，轮转没在跑")
    if len(arch) > keep:
        problems.append(f"留档 {len(arch)} 份 > 上限 {keep}")
    detail = (f"{os.path.basename(path)} 当前 {cur} 字节 / {rows_of(path) or 0} 行，"
              f"轮转留档 {len(arch)} 份，合计 {tot} 字节（硬上限 {max_bytes}×{keep + 1}）")
    return problems, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8310)
    ap.add_argument("--json", default=os.path.join(HERE, "scratch", "health_probe.json"))
    ap.add_argument("--skip-coverage", action="store_true")
    ap.add_argument("--heavy-provenance", action="store_true",
                    help="额外做一次抽样重嵌对齐验证（要加载模型，几十秒）")
    a = ap.parse_args()
    res = []

    # 1) 监听面：只允许回环，且只能有一个实例
    rows, err = listeners(a.port)
    if err:
        res.append(check("listener", "WARN", err))
    elif not rows:
        res.append(check("listener", "FAIL", f"端口 {a.port} 无监听——服务没起来"))
    else:
        bad = [r for r in rows if not r[0].startswith("127.0.0.1")]
        pids = sorted({r[1] for r in rows})
        if bad:
            res.append(check("listener", "FAIL", f"非回环地址在监听 {bad}（个人服务不应对外）"))
        elif len(pids) > 1:
            res.append(check("listener", "FAIL", f"多个实例共用端口 {pids}——检索结果会随进程漂移"))
        else:
            res.append(check("listener", "PASS", f"127.0.0.1:{a.port} 单实例 pid={pids[0]}"))

    # 2) /health
    try:
        h = get(a.port, "/health")
        if not h.get("ok"):
            res.append(check("health", "FAIL", f"ok=false error={h.get('error')}"))
        elif h.get("error"):
            res.append(check("health", "FAIL", f"error={h['error']}"))
        elif h.get("warn"):
            res.append(check("health", "WARN", f"warn={h['warn']}"))
        else:
            res.append(check("health", "PASS", f"ready，mem/zh/comp 挂载={h.get('mem')}/{h.get('zh')}/{h.get('comp')}"))
        health = h
    except (urllib.error.URLError, OSError, ValueError) as e:
        res.append(check("health", "FAIL", f"/health 取不到: {e}"))
        health = None

    # 3) /stats：内存态 vs 磁盘态、过期区、最近热重载
    stats = None
    if health:
        try:
            stats = get(a.port, "/stats")
        except (urllib.error.URLError, OSError, ValueError) as e:
            res.append(check("stats", "FAIL", f"/stats 取不到: {e}"))
    if stats:
        drift = [z for z, v in (stats.get("disk", {}).get("drift") or {}).items() if v]
        res.append(check("loaded_vs_disk", "FAIL" if drift else "PASS",
                         f"内存态≠磁盘态的区: {drift}" if drift else
                         "五区内存块数与磁盘索引一致 " + json.dumps(stats.get("zones"), ensure_ascii=False)))
        st = stats.get("stale") or []
        res.append(check("stale_indexed", "FAIL" if st else "PASS",
                         f"已入库文件比索引新（需重建/热重载）: {json.dumps(st, ensure_ascii=False)}"
                         if st else "已入库文件全部不晚于索引"))
        rl = stats.get("reload") or {}
        bad_rl = {z: v for z, v in rl.items() if isinstance(v, str) and ("失败" in v or "error" in v.lower())}
        res.append(check("reload_last", "FAIL" if bad_rl else "PASS",
                         f"最近热重载失败: {bad_rl}" if bad_rl else f"最近热重载: {json.dumps(rl, ensure_ascii=False)[:120]}"))
        res.append(check("uptime", "PASS" if stats.get("uptime_s") else "WARN",
                         f"uptime {stats.get('uptime_s')}s，本进程查询 {stats.get('perf', {}).get('n')} 次，"
                         f"p95 {stats.get('perf', {}).get('p95_ms')}ms"))

    # 3b) C 组：语料比索引新（add 写完 md 就崩的那一段窗口）。只报警，重建闸门是人走的。
    if health:
        lag = health.get("lag")
        if lag is None:
            res.append(check("corpus_lag", "WARN",
                             "/health 没有 lag 字段——服务在跑 C 之前的代码（见 code_freshness）"))
        else:
            res.append(check("corpus_lag", "WARN" if lag else "PASS",
                             f"索引落后语料，需人工重建该区: {json.dumps(lag, ensure_ascii=False)}"
                             if lag else "mem/epi 语料不晚于索引"))

    # 4) 溯源：五区都要有 sidecar，且描述的就是当前这份 chunks
    for z, (ck, vc) in ZONE_FILES.items():
        if not os.path.exists(ck):
            res.append(check(f"provenance[{z}]", "WARN", f"{ck} 不存在（该区未启用）"))
            continue
        n = rows_of(ck)
        p = vc + ".provenance.json"
        if not os.path.exists(p):
            res.append(check(f"provenance[{z}]", "FAIL", f"{os.path.basename(vc)} 缺溯源 sidecar"))
            continue
        try:
            prov = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError) as e:
            res.append(check(f"provenance[{z}]", "FAIL", f"sidecar 读不了: {e}"))
            continue
        want_sha = hashlib.sha256(open(ck, "rb").read()).hexdigest()[:16]
        if prov.get("model") != EXPECT_MODEL[z]:
            res.append(check(f"provenance[{z}]", "FAIL",
                             f"模型 {prov.get('model')} ≠ 配置 {EXPECT_MODEL[z]}"))
        elif prov.get("n_chunks") != n:
            res.append(check(f"provenance[{z}]", "FAIL",
                             f"溯源 n_chunks={prov.get('n_chunks')} ≠ 现索引 {n}（vec 是另一代的）"))
        elif prov.get("chunks_sha256") != want_sha:
            res.append(check(f"provenance[{z}]", "WARN",
                             f"溯源块哈希 {prov.get('chunks_sha256')} ≠ 现档 {want_sha}"
                             f"（索引重建过但向量未刷新；跑 kb/write_provenance.py --zone {z}）"))
        else:
            res.append(check(f"provenance[{z}]", "PASS",
                             f"{prov.get('model')} / {n} 块 / sha {want_sha} 一致"))
    if a.heavy_provenance:
        import numpy as np
        import write_provenance as WP
        for z, (ck, vc) in ZONE_FILES.items():
            if not os.path.exists(vc):
                continue
            chunks = [json.loads(l) for l in open(ck, encoding="utf-8") if l.strip()]
            vec = np.load(vc)
            _b, mean, mn, idx = WP.verify(chunks, vec, WP.sample_rows(len(chunks), 6), EXPECT_MODEL[z])
            res.append(check(f"vec_alignment[{z}]", "PASS" if mn >= 0.999 else "FAIL",
                             f"抽样 {len(idx)} 行重嵌余弦 mean={mean:.4f} min={mn:.4f}"))

    # 5) 进程：代码是否比进程新、内存与线程
    pid = None
    if rows:
        pid = rows[0][1]
        info = proc_info(pid)
        if not info:
            res.append(check("process", "WARN", f"pid={pid} 取不到进程信息"))
        else:
            pm = round(info["pm"] / 1048576)
            res.append(check("process", "WARN" if pm > PRIVATE_MB_WARN or info["th"] > THREADS_WARN else "PASS",
                             f"pid={pid} 私有内存 {pm}MB（阈值 {PRIVATE_MB_WARN}）· 线程 {info['th']}（阈值 {THREADS_WARN}）"
                             f" · 启动 {info['st']}"))
            serve = os.path.join(HERE, "kb_serve.py")
            code_ts = datetime.datetime.fromtimestamp(os.path.getmtime(serve)).strftime("%Y-%m-%d %H:%M:%S")
            if code_ts > info["st"]:
                res.append(check("code_freshness", "FAIL",
                                 f"kb_serve.py 改于 {code_ts}，晚于进程启动 {info['st']}——服务在跑旧代码，需重启"))
            else:
                res.append(check("code_freshness", "PASS", f"进程启动 {info['st']} 晚于 kb_serve.py {code_ts}"))

    # 6) 冒烟查询：四区各来一发，看返回与延迟
    #    先每档发一发不计时的探针：模型会话首查要付 ONNX warm-up（实测刚启动时
    #    zh 首查 6066ms，预热后 58ms），不预热会把健康服务误报成延迟超标。
    for kb in {k for k, _, _ in SMOKE}:
        try:
            get(a.port, "/search", {"query": "预热", "k": 1, "kb": kb}, timeout=120)
        except (urllib.error.URLError, OSError, ValueError):
            pass
    lats = []
    for kb, q, k in SMOKE:
        t0 = time.time()
        try:
            rows = get(a.port, "/search", {"query": q, "k": k, "kb": kb}, timeout=60)
            ms = int((time.time() - t0) * 1000)
            lats.append(ms)
            # /search 返回裸列表；出错时是 {"error": ...} 字典
            if isinstance(rows, dict):
                res.append(check(f"search[{kb}]", "FAIL", f"“{q}” 返回错误: {rows.get('error')}"))
            elif not rows:
                res.append(check(f"search[{kb}]", "FAIL", f"“{q}” 零命中"))
        except (urllib.error.URLError, OSError, ValueError) as e:
            res.append(check(f"search[{kb}]", "FAIL", f"“{q}” 请求失败: {e}"))
    if lats:
        worst = max(lats)
        res.append(check("search_latency", "WARN" if worst > LAT_WARN_MS else "PASS",
                         f"冒烟延迟 {lats}ms（阈值 {LAT_WARN_MS}）"))
    # mem/epi 是否附在主库结果后
    if health and health.get("mem"):
        try:
            rows = get(a.port, "/search", {"query": "海马体 记忆 演练", "k": 3}, timeout=60)
            zones = {x.get("zone") for x in rows} if isinstance(rows, list) else set()
            res.append(check("tail_zones", "PASS" if {"mem", "epi"} & zones else "WARN",
                             f"主库查询附带的尾部区: {sorted(z for z in zones if z != 'doc') or '无'}"))
        except (urllib.error.URLError, OSError, ValueError) as e:
            res.append(check("tail_zones", "FAIL", str(e)))

    # 7) 占盘增长：三份只增档各自的轮转是否真在跑（阈值取自 kb_serve，不另抄）
    try:
        import kb_serve as SV
    except Exception as e:                                        # noqa: BLE001
        SV = None
        res.append(check("rotation_config", "FAIL", f"读不到 kb_serve 的轮转阈值: {type(e).__name__}: {e}"))
    rows_h = []
    if SV:
        for label, p, mb, kp, miss_note in (
                ("query_log_rotation", os.path.join(HERE, "query_log.jsonl"),
                 SV.LOG_MAX_BYTES, SV.LOG_KEEP, "本进程还没写过查询日志"),
                ("reload_history", SV.RELOAD_HISTORY,
                 SV.RELOAD_HIST_MAX_BYTES, SV.RELOAD_HIST_KEEP,
                 "服务在跑 A③ 之前的代码，或起后从未重载过任何区")):
            if not os.path.exists(p) and not archives_of(p):
                res.append(check(label, "WARN", f"{os.path.basename(p)} 不存在（{miss_note}）"))
                continue
            bad, detail = disk_budget(p, mb, kp)
            res.append(check(label, "FAIL" if bad else "PASS",
                             detail + "；" + "；".join(bad) if bad else detail))
        rows_h = []
        if os.path.exists(SV.RELOAD_HISTORY):
            try:
                rows_h = [json.loads(l) for l in open(SV.RELOAD_HISTORY, encoding="utf-8") if l.strip()]
            except ValueError as e:
                res.append(check("reload_history_last", "FAIL", f"档读不了: {e}"))
        if not rows_h:
            arch = archives_of(SV.RELOAD_HISTORY)
            if arch:                       # 刚轮转完当前档是空的，最近一次结果在最新的留档里
                try:
                    rows_h = [json.loads(l) for l in
                              open(os.path.join(HERE, arch[-1]), encoding="utf-8") if l.strip()]
                except (OSError, ValueError):
                    rows_h = []
        if rows_h:
            # "未知区 xxx"是调用方打错的请求，不是索引坏了——别让它把整个探测按在 FAIL 上。
            # 真正的构建失败（区还在内存里但磁盘已换）才是 FAIL 的对象。
            fails_h = [r for r in rows_h if not r.get("ok") and "未知区" not in (r.get("message") or "")]
            typo_n = sum(1 for r in rows_h if not r.get("ok") and "未知区" in (r.get("message") or ""))
            last = rows_h[-1]
            msg = f"最近一条 {last.get('ts')} {last.get('zone')} ok={last.get('ok')}：{last.get('message')}"
            if typo_n:
                msg += f"；另有 {typo_n} 条请求区名打错（不算索引故障）"
            if fails_h:
                f = fails_h[-1]
                msg += f"；构建失败 {len(fails_h)} 条，最近 {f.get('ts')} {f.get('zone')}：{f.get('message')}"
            res.append(check("reload_history_last", "FAIL" if not last.get("ok") and fails_h else
                             "WARN" if fails_h else "PASS", msg))

    # 服务日志跨重启留档：_start.ps1 起来前把上一程的 out/err 归档（不再截断，崩溃现场才拿得回）
    for fn in ("serve.out.log", "serve.err.log"):
        p = os.path.join(HERE, fn)
        arch = archives_of(p)
        if not arch:
            res.append(check(f"serve_log[{fn}]", "WARN",
                             f"{fn} 没有 .YYYYmmdd-HHMMSS 留档——新 _start.ps1 还没走过一次，"
                             f"上一程的现场仍会被截断"))
        else:
            res.append(check(f"serve_log[{fn}]", "PASS",
                             f"{fn} 跨重启留档 {len(arch)} 份，最近 {arch[-1]}"
                             f"（当前档 {os.path.getsize(p) if os.path.exists(p) else 0} 字节）"))

    # 每个区现在是不是还卡在失败上（区名打错是调用方的事，不算区坏）。
    latest_ok = {}
    for r in rows_h:
        if "未知区" not in (r.get("message") or ""):
            latest_ok[r.get("zone")] = bool(r.get("ok"))
    unrecovered = sorted(z for z, v in latest_ok.items() if not v)

    for fn in ("serve.err.log", "serve.out.log"):
        p = os.path.join(HERE, fn)
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            continue
        txt = open(p, encoding="utf-8", errors="replace").read()
        fatal = [l for l in txt.splitlines() if "FATAL" in l or "BOOT FAILED" in l]
        rl = [l for l in txt.splitlines() if "RELOAD FAILED" in l]
        if fatal:
            res.append(check("serve_err", "FAIL", f"{fn} 有致命记录: {fatal[-1][:160]}"))
        elif rl and unrecovered:
            res.append(check("serve_err", "FAIL",
                             f"{fn} 有 {len(rl)} 条重载失败且这些区仍未恢复 {unrecovered}"
                             f"（内存里还是旧档）: {rl[-1][:140]}"))
        elif rl:
            # A③ 之后"err.log 里有 traceback"本身不是故障：watcher 撞见 chunks 已换、
            # bm25 未换的中间态会拒档保留旧副本，重建齐了自动追平——现场要留着，灯要放绿。
            res.append(check("serve_err", "WARN",
                             f"{fn} 记有 {len(rl)} 条重载失败，但这些区后来都重载成功了"
                             f"（中间态被行数断言挡住、旧档保留，属 A③ 设计留档）: {rl[-1][:140]}"))

    # 8) 覆盖审计（正式文档写了但没入库，绿灯看不见）
    if not a.skip_coverage:
        try:
            import coverage_audit as CA
            rep = {}
            for zone, zc in CA.ZONES.items():
                if not os.path.exists(zc["index"]):
                    continue
                _rows, indexed = CA.read_index(zc["index"])
                disk = CA.disk_files(zc)
                miss = sorted(disk - set(indexed))
                formal = [f for f in miss if not CA.is_temp(f)]
                temp = [f for f in miss if CA.is_temp(f)]
                if formal or temp:
                    res.append(check(f"coverage[{zone}]", "WARN",
                                     f"未入库 正式 {len(formal)} {formal[:4]} / 临时件 {len(temp)}"
                                     f"（重建会带进候选池）"))
            if not any(r["check"].startswith("coverage") for r in res):
                res.append(check("coverage", "PASS", "五区没有未入库文件"))
        except Exception as e:                                    # noqa: BLE001
            res.append(check("coverage", "WARN", f"覆盖审计没跑成: {type(e).__name__}: {e}"))

    fails = [r for r in res if r["status"] == "FAIL"]
    warns = [r for r in res if r["status"] == "WARN"]
    print(f"海马体健康探测 {datetime.datetime.now():%Y-%m-%d %H:%M:%S} "
          f"port={a.port} → {'FAIL' if fails else 'PASS'}（{len(fails)} FAIL / {len(warns)} WARN）")
    for r in res:
        print(f"  [{r['status']:4s}] {r['check']}: {r['detail']}")
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    json.dump({"at": datetime.datetime.now().isoformat(timespec="seconds"), "port": a.port,
               "pid": pid, "fail": len(fails), "warn": len(warns), "checks": res},
              open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("json →", a.json)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
