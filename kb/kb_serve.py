"""kb_serve.py — 海马体常驻检索服务（127.0.0.1:8310）。
模型只加载一次（主库 doc+mem 共用 bge-m3 ONNX 实例），之后每次查询毫秒级。
  GET  /health              -> {"ok":true,...}
  POST /search {"query":"..","k":8,"mode":"hybrid-v2","layout":"tiered","kb":"main|zh|comp"}
默认档位：main=bge-m3 + hybrid-v2（min-max 0.5:0.5，55 题 81.8% 基线）；
         zh=e5-large + hybrid(RRF)（专利库验证配置，勿用 hybrid-v2）；
         comp=bge-m3 + hybrid-v2（竞品域 COMP-v2 基线 83.7%）。
热重载：POST /reload {"zone":"main|mem|epi|zh|comp|all"} 原子重载该区（复用共享嵌入器、
失败保留旧档）；后台每 10s 监视各区 chunks.jsonl 与 alias_map.json，变更自动重载。
每次重载（含自动的、成功与失败都算）落 kb/reload_history.jsonl，失败还带 traceback——
/stats 与 /health 的 reload_history / reload_fail 字段就是读它，进程重启后照样查得到上次坏在哪。
日志占盘有硬上限：query_log.jsonl 超过 KB_LOG_MAX_MB（默认 10MB）就轮转，留 KB_LOG_KEEP（默认 3）份；
reload_history.jsonl 同理（KB_RELOAD_HIST_MAX_MB 默认 2MB × 留 3 份），且同区同因的连续失败只记一次。
watcher 另报"语料比索引新"（corpus_lag）：崩溃打断 add/入库后没人修，这里只报警、不自动重建。
端口被占用时明确报错并非零退出（已有实例不受影响）。e5 懒加载（zh 首查才加载模型）。
"""
import json, os, sys, time, datetime, threading, traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kb_log_archive
MODEL_MAIN = os.environ.get("KB_MODEL", "bge-m3-onnx")            # 与 fuse.py 默认一致
MODEL_ZH = "intfloat/multilingual-e5-large"                        # 专利库验证过的模型
DEFAULT_MODE = {"main": "hybrid-v2", "zh": "hybrid", "comp": "hybrid-v2"}
MODES_OK = ("hybrid", "hybrid-v2", "bm25", "vec", "hybrid-r", "bm25-r", "vec-r")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
PORT = int(os.environ.get("KB_PORT", "8310"))
LOG = os.path.join(HERE, "query_log.jsonl")
RELOAD_HISTORY = os.path.join(HERE, "reload_history.jsonl")
# query_log 每查询一行、无上限增长；健康探测只在 50MB 时报警，报完还在长。
# 这里直接压在阈值内：KB_LOG_MAX_MB×(KB_LOG_KEEP+1) 就是它占盘的硬上限（默认 40MB）。
LOG_MAX_BYTES = int(float(os.environ.get("KB_LOG_MAX_MB", "10")) * 1024 * 1024)
LOG_KEEP = int(os.environ.get("KB_LOG_KEEP", "3"))
# reload_history 也是只增档，且失败时带 traceback（几 KB 一条）。它比 query_log 小得多，
# 但 watcher 会重试，所以照样要有上限——MB 级足够留下几个月的现场。
RELOAD_HIST_MAX_BYTES = int(float(os.environ.get("KB_RELOAD_HIST_MAX_MB", "2")) * 1024 * 1024)
RELOAD_HIST_KEEP = int(os.environ.get("KB_RELOAD_HIST_KEEP", "3"))
_log_lock = threading.Lock()     # 只保护查询日志追加，不包检索——检索本身无共享可变状态
_hist_lock = threading.Lock()    # reload_history 的追加 + 轮转 + 去重键，同一把锁内完成
_hits_lock = threading.Lock()
HITS = {}                        # (zone, file, line) -> [被返回次数, heading]（进程内累计）
ZONES = ("main", "mem", "epi", "zh", "comp")

STATE = {"ready": False, "main": None, "zh": None, "comp": None, "error": "", "warn": [], "lag": []}
T0 = time.time()
LATS = deque(maxlen=200)          # 本次进程生命周期内的查询延迟样本
_FRESH = {}                       # zone -> {"ts": 上次扫描时间, "mtime": 最新已入库文件, "path": 该文件}
_IDX_FILES = {}                   # zone -> {"mtime": 索引档时间, "ts", "files": 索引内文件集合}
WORKSPACE = os.path.dirname(os.path.dirname(BASE))   # HERE=…/kb, BASE=…/hippocampus -> <本地工作区>
# load_kb 会临时改 KB_DIR/KB_VEC/KB_MODEL 并 importlib.reload 模块（全局状态），
# 所有构建调用（boot 与 /reload）必须串行。检索路径不读这三个 env、KB 实例构造后自包含，
# 所以查询全程不加这把锁——不会把查询串行化。
_load_lock = threading.Lock()
RELOAD = {"last": {}, "last_fail": None, "watch": True}   # zone -> 最近一次重载结果（面板可见）
_HIST_LAST_FAIL = {}     # zone -> 已落盘的同因失败文案；连续重复不再追加，该区一成功即清除

def log_query(rec):
    with _log_lock:
        try:
            over = os.path.getsize(LOG) > LOG_MAX_BYTES
        except OSError:
            over = False
        if over:
            # 轮转与追加共用这把锁：改名不会踩到别的写者的句柄，也不会两个线程都判定超限
            kb_log_archive.rotate(LOG, LOG_KEEP)
            print(f"[kb_serve] query_log 超 {LOG_MAX_BYTES / 1048576:.1f}MB，已轮转（留 {LOG_KEEP} 份）",
                  flush=True)
        with open(LOG, "a", encoding="utf-8") as w:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

def zone_root(zone):
    """各区语料根（chunks 里的 file 相对它）。主库是全工作区但排除 research/。"""
    return {"main": WORKSPACE,
            "mem": os.path.join(HERE, "mem", "corpus"),
            "epi": os.path.join(HERE, "epi", "corpus"),
            "zh": os.path.join(BASE, "corpus_zh"),
            "comp": os.path.join(BASE, "corpus_comp", "md")}.get(zone)

def zone_index(zone):
    return {"main": os.path.join(HERE, "chunks.jsonl"),
            "mem": os.path.join(HERE, "mem", "chunks.jsonl"),
            "epi": os.path.join(HERE, "epi", "chunks.jsonl"),
            "zh": os.path.join(BASE, "kb_zh", "chunks.jsonl"),
            "comp": os.path.join(BASE, "kb_comp", "chunks.jsonl")}[zone]

def indexed_files(zone):
    """索引档里实际出现的文件集合（按索引档 mtime 缓存 30s）。
    stale 只以这些文件为准：新写但没入库的文件（如 agents.md 规程）不该把主库灯点亮。"""
    p = zone_index(zone)
    try:
        im = os.path.getmtime(p)
    except OSError:
        return None
    now = time.time()
    c = _IDX_FILES.get(zone)
    if c and c["mtime"] == im and now - c["ts"] <= 30:
        return c["files"]
    files = set()
    try:
        with open(p, encoding="utf-8") as r:
            for line in r:
                try:
                    files.add(json.loads(line)["file"].replace("/", os.sep))
                except (ValueError, KeyError):
                    continue
    except OSError:
        return None
    _IDX_FILES[zone] = {"mtime": im, "ts": now, "files": files}
    return files

def corpus_fresh(zone):
    """该区**已入库**文件里最新的修改时间。按区缓存 30s。
    旧实现扫整个语料目录：工作区里新写而没入库的 md（规程、笔记）会把主库长期误判 stale。"""
    now = time.time()
    c = _FRESH.get(zone)
    if c and now - c["ts"] <= 30:
        return c
    best, bp = 0.0, ""
    root = zone_root(zone)
    files = indexed_files(zone)
    if root and files:
        for f in files:
            p = os.path.join(root, f)
            try:
                m = os.path.getmtime(p)
            except OSError:
                continue
            if m > best:
                best, bp = m, p
    c = {"ts": now, "mtime": best, "path": bp}
    _FRESH[zone] = c
    return c

def tail_log(n=20):
    try:
        with open(LOG, encoding="utf-8") as r:
            lines = r.readlines()[-n:]
        return [json.loads(l) for l in reversed(lines)]
    except OSError:
        return []

def reload_history_tail(n=10):
    """最近 n 次重载（含轮询自动重载），新的在前。进程死了也查得到——这就是落盘的意义。"""
    try:
        with open(RELOAD_HISTORY, encoding="utf-8") as r:
            lines = r.readlines()[-n:]
        return [json.loads(l) for l in reversed(lines)]
    except (OSError, ValueError):
        return []

def mem_entries():
    out = []
    d = os.path.join(HERE, "mem", "corpus")
    if not os.path.isdir(d):
        return out
    fm = __import__("re").compile(r"^---\n(.*?)\n---", __import__("re").S)
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".md"):
            continue
        head = open(os.path.join(d, fn), encoding="utf-8").read(1500)
        meta = {"file": fn, "confidence": None, "superseded_by": "", "created": ""}
        m = fm.match(head)
        if m:
            for line in m.group(1).splitlines():
                k = line.split(":", 1)[0].strip()
                if k in meta:
                    meta[k] = line.split(":", 1)[1].strip()
        out.append(meta)
    return out

def mtime(p):
    try: return os.path.getmtime(p)
    except OSError: return 0

def disk_chunk_count(path):
    try:
        with open(path, encoding="utf-8") as r:
            return sum(1 for l in r if l.strip())
    except OSError:
        return None

def stats_json():
    f, zh, comp = STATE["main"], STATE["zh"], STATE.get("comp")
    tails = {t["name"]: len(t["kb"].chunks) for t in (f.tails if f else [])}
    idx = {
        "main_chunks": len(f.doc.chunks) if f else 0,
        "mem_chunks": tails.get("mem", 0),
        "epi_chunks": tails.get("epi", 0),
        "zh_chunks": len(zh.doc.chunks) if zh else 0,
        "comp_chunks": len(comp.doc.chunks) if comp else 0,
    }
    # 内存态 vs 磁盘态：自动监视通常 ~10s 内追上磁盘。drift 长期存在说明热重载被
    # 行数/模型校验挡住（见 reload 字段），需人工确认磁盘那份是过闸的档位。
    disk = {z: disk_chunk_count(zone_index(z)) for z in ZONES}
    drift = {z: disk[z] is not None and disk[z] != idx[f"{z}_chunks"] for z in ZONES}
    index_files = {z: {"path": os.path.relpath(zone_index(z), BASE), "mtime": fmt_ts(mtime(zone_index(z)))}
                   for z in ZONES}
    # 语料新鲜度按区判定：各区语料目录不同，拿主库的索引时间去比专利库语料没有意义
    corpus = {}
    for z in ZONES:
        cf = corpus_fresh(z)
        root = zone_root(z)
        corpus[z] = {"latest_mtime": fmt_ts(cf["mtime"]),
                     "latest_file": os.path.relpath(cf["path"], root) if cf["path"] and root else "",
                     "index_mtime": fmt_ts(mtime(zone_index(z))),
                     "stale": cf["mtime"] > mtime(zone_index(z))}
    stale_zones = [z for z in ZONES if corpus[z]["stale"]]
    lats = sorted(LATS)
    mem = mem_entries()
    with _hits_lock:
        top = sorted(HITS.items(), key=lambda kv: -kv[1][0])[:15]
        nhit = sum(v[0] for v in HITS.values())
        hits = [{"zone": k[0], "file": k[1], "line": k[2], "heading": v[1], "count": v[0]}
                for k, v in top]
    return {
        "ok": STATE["ready"], "uptime_s": round(time.time() - T0),
        "error": STATE["error"] or None,
        "perf": {"n": len(lats), "avg_ms": round(sum(lats) / len(lats)) if lats else None,
                 "p95_ms": lats[int(len(lats) * 0.95) - 1] if lats else None},
        "zones": idx,
        "disk": dict(disk, drift=drift),
        "index_files": index_files,
        "corpus": corpus,
        "stale": stale_zones,
        "reload": RELOAD["last"],
        "reload_history": reload_history_tail(10),
        "lag": STATE["lag"],
        "hits": {"total": nhit, "top": hits, "boost": False,
                 "note": "命中计数只做展示；排序加成未过闸门，默认关闭"},
        "mem": {"total": len(mem),
                "superseded": sum(1 for x in mem if x["superseded_by"]),
                "low_conf": sum(1 for x in mem if x["confidence"] and float(x["confidence"]) < 0.5),
                "entries": mem},
        "recent": tail_log(20),
    }


def fmt_ts(v):
    return datetime.datetime.fromtimestamp(v).strftime("%m-%d %H:%M") if v else "-"

def check_provenance(zone, doc_dir, vec_name, model):
    """启动/重载护栏：vec provenance 的模型名须与该库配置一致。
    行数对齐由 KB.__init__ 断言兜底；这里防的是"形状合法但模型不对"的跨进程换档
    （Hermes 侧与本服务共用磁盘索引，一边换模型另一边不能静默受害）。
    返回错误字符串或 None。provenance 缺失不强制（mem/epi 历史档没有）。"""
    p = os.path.join(doc_dir, vec_name + ".provenance.json")
    if not os.path.exists(p):
        return None
    try:
        prov = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return f"{zone}: provenance 读取失败 {os.path.basename(p)}: {e}"
    pm = prov.get("model")
    if pm and pm != model:
        return f"{zone}: 向量档模型 {pm} ≠ 配置 {model}，拒绝挂载（{os.path.basename(p)}）"
    return None

def tail_prov_warnings():
    """尾部区（mem/epi）溯源体检：缺 sidecar 不拦（历史档本就无），模型名不符就报警。
    与主库的致命校验分级不同——尾部只附在主库结果后面，一份坏尾部档不该让整服务起不来，
    但也不能静默：两边共用磁盘索引，Hermes 侧把 mem/epi 换成别的模型，这里必须看见。"""
    out = []
    for z in ("mem", "epi"):
        d = os.path.join(HERE, z)
        if os.path.exists(os.path.join(d, "chunks.jsonl")):
            w = check_provenance(z, d, "vec.npy", MODEL_MAIN)
            if w:
                out.append(w)
    return out

# C 组：只查尾部两区。主库/专利/竞品故意不查——工作区里新写而没入库的笔记是常态，
# 报出来只会变成永久噪声（和 /stats 的 stale 口径一致：只比索引里实际出现的文件）。
LAG_ZONES = ("mem", "epi")
LAG_EVERY_SEC = 60

def corpus_files(zone):
    """该区语料目录里的文档文件名集合（mem/epi 是平铺小目录，直接列一层）。"""
    root = zone_root(zone)
    if not root or not os.path.isdir(root):
        return set()
    return {f for f in os.listdir(root) if f.lower().endswith((".md", ".txt"))}

def unindexed_files(zone):
    """语料里有、索引档里没有的文件；索引档整体缺失时返回全部语料文件。"""
    files = corpus_files(zone)
    if not files:
        return []
    idx = indexed_files(zone)
    if idx is None:
        return sorted(files)
    return sorted(f for f in files if f not in idx)

def corpus_lag_warnings():
    """C 组报警：写完 md 就崩的形态。kb_remember add / kb_epilogue ingest 是两步（先落语料、
    再重建索引），中间死掉的话索引档 mtime 没变，watcher 永远看不见，那条记忆/那段复盘就静静
    躺在语料里查不到。这里只把它报出来——**不自动重建**：尾部区的入库闸门是人走的，
    自动重建等于把写权限交给 watcher。"""
    out = []
    for z in LAG_ZONES:
        if not os.path.isdir(os.path.join(HERE, z, "corpus")):
            continue
        miss = unindexed_files(z)
        if not miss:
            continue
        named = "、".join(miss[:3]) + ("…" if len(miss) > 3 else "")
        why = "索引档不存在" if not os.path.exists(zone_index(z)) else "索引落后语料"
        out.append(f"{z} 语料有 {len(miss)} 个文件未入库（{named}）——{why}，"
                   f"需人工重建该区（重建后 POST /reload 或等自动监视追上）")
    return out

def refresh_lag_alerts():
    """刷新 STATE["lag"]，并把它并进 STATE["warn"]（现有健康探测看 warn，藏在旁边就看不见）。
    同一条报警只出现一次；语料追上索引后撤销，不留长黄的灯。"""
    lag = corpus_lag_warnings()
    for gone in [w for w in STATE["lag"] if w not in lag]:
        if gone in STATE["warn"]:
            STATE["warn"].remove(gone)
    STATE["lag"] = lag
    for w in lag:
        if w not in STATE["warn"]:
            STATE["warn"].append(w)
            print(f"[kb_serve] WARN: {w}", file=sys.stderr, flush=True)
    return lag

def _share_m3(f, embedder):
    """把已加载的 m3 实例复用给新 Fused 的 doc 与全部尾部区，不重新加载模型。"""
    if embedder is None:
        return
    f.doc._embedder = embedder
    for t in f.tails:
        t["kb"]._embedder = embedder

def _build_main(embedder):
    from fuse import Fused
    f = Fused(model=MODEL_MAIN)      # 含 mem/epi 尾部区
    _share_m3(f, embedder)
    return f

def _build_zh(embedder):
    from fuse import Fused
    zh = os.path.join(BASE, "kb_zh")
    old = os.environ.get("KB_VENAME")
    os.environ["KB_VENAME"] = "vec.npy"
    try:
        new = Fused(model=MODEL_ZH, doc_dir=zh, mem_dir=None)
    finally:
        if old is None:
            os.environ.pop("KB_VENAME", None)
        else:
            os.environ["KB_VENAME"] = old
    if embedder is not None:
        new.doc._embedder = embedder  # e5 已加载过就复用，否则保持懒加载
    return new

def _build_comp(embedder):
    from fuse import Fused
    comp = os.path.join(BASE, "kb_comp")
    old = os.environ.get("KB_VENAME")
    os.environ["KB_VENAME"] = "vec.npy"
    try:
        c = Fused(model=MODEL_MAIN, doc_dir=comp, mem_dir=None)
    finally:
        if old is None:
            os.environ.pop("KB_VENAME", None)
        else:
            os.environ["KB_VENAME"] = old
    if embedder is not None:
        c.doc._embedder = embedder
    return c

def _reload_tail(f, name, embedder):
    """只重建 main 的某个尾部区（mem/epi），doc 不动；tails 列表整体原子替换。"""
    from fuse import load_kb, mem_flags
    d = os.path.join(HERE, name)
    kb, mod = load_kb(d, MODEL_MAIN, os.path.join(d, "vec.npy"))
    if embedder is not None:
        kb._embedder = embedder
    entry = {"name": name, "kb": kb, "mod": mod, "flags": mem_flags(os.path.join(d, "corpus"))}
    tails, found = [], False
    for t in f.tails:
        if t["name"] == name:
            tails.append(entry); found = True
        else:
            tails.append(t)
    if not found:
        tails.append(entry)          # epi 首次 PROMOTED 后补挂
    f.tails = tails
    f.has_mem = any(t["name"] == "mem" for t in tails)

def reload_zone(zone):
    """原子重载一个区：新实例构建成功（含 chunks/bm25/vec 行数断言）才替换，失败保留旧档。
    调用方须持有 _load_lock。返回 (ok, message, traceback_text|None)——
    traceback 只在异常分支非空：保住旧档是运行时行为，但事后要查得出是哪一步炸的。"""
    try:
        main = STATE.get("main")
        m3 = main.doc._embedder if main is not None else None
        if zone == "main":
            STATE["main"] = _build_main(m3)
        elif zone in ("mem", "epi"):
            if main is None:
                return False, "main 未就绪", None
            _reload_tail(main, zone, m3)
        elif zone == "zh":
            old = STATE.get("zh")
            STATE["zh"] = _build_zh(old.doc._embedder if old is not None else None)
        elif zone == "comp":
            STATE["comp"] = _build_comp(m3)
        elif zone == "all":
            STATE["main"] = _build_main(m3)
            old = STATE.get("zh")
            STATE["zh"] = _build_zh(old.doc._embedder if old is not None else None)
            STATE["comp"] = _build_comp(STATE["main"].doc._embedder)
        else:
            return False, f"未知区 {zone}（可选 main|mem|epi|zh|comp|all）", None
        return True, f"{zone} 重载完成", None
    except Exception as e:
        return False, f"{zone} 重载失败，已保留旧档: {type(e).__name__}: {e}", traceback.format_exc()

def record_reload(zone, ok, message, tb=None):
    """重载结果落盘。/stats 的 reload 字段与进程同生共死，重启后"上一次到底哪区坏了"
    就查无对证；崩溃/重启后的第一轮排查恰恰最需要这条。

    两处自我约束，否则这份档会变成新的增长源：
    ① 同区同文案的连续失败只记一次——watcher 对失败的区每 10s 重试且基线不更新（这是
       设计：三件齐了才切换），一天可写 8640 条同样的 traceback。该区一旦重载成功就清掉
       去重键，"坏过→修好→又坏"要重新记账。
    ② 超过 RELOAD_HIST_MAX_BYTES 就轮转（与 query_log 同一套 kb_log_archive），留
       RELOAD_HIST_KEEP 份。
    /health 的 reload_fail 不受①影响，每次都刷新，所以"现在还在失败"看得到。
    返回值 = 这条有没有真的落盘，stderr 那一句用它一起消重（serve.err.log 现在只归档
    不截断，不去重的话同一个坏区每 10s 写一段 traceback，增长源只是换了个文件）。"""
    row = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
           "zone": zone, "ok": bool(ok), "message": message}
    if tb:
        row["traceback"] = tb
    if not ok:
        RELOAD["last_fail"] = row
    try:
        with _hist_lock:
            if ok:
                _HIST_LAST_FAIL.pop(zone, None)
            elif _HIST_LAST_FAIL.get(zone) == message:
                return False
            else:
                _HIST_LAST_FAIL[zone] = message
            try:
                over = os.path.getsize(RELOAD_HISTORY) > RELOAD_HIST_MAX_BYTES
            except OSError:
                over = False      # 首轮还没有这份档，不算超限
            if over:
                # 先轮转再追加（和 log_query 同一顺序）：轮转是改名，追加会顺手把当前档
                # 重新创建，读侧永远不会遇到"档不存在"的空窗。
                kb_log_archive.rotate(RELOAD_HISTORY, RELOAD_HIST_KEEP)
                print(f"[kb_serve] reload_history 超 {RELOAD_HIST_MAX_BYTES / 1048576:.1f}MB，"
                      f"已轮转（留 {RELOAD_HIST_KEEP} 份）", flush=True)
            with open(RELOAD_HISTORY, "a", encoding="utf-8") as w:
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
            return True
    except OSError as e:
        print(f"[kb_serve] reload_history 写入失败（不影响检索）: {e}", file=sys.stderr, flush=True)
        return True     # 写不进去也当作"记过了"，否则 stderr 会跟着无限刷

def reload_and_record(zone):
    """重载的唯一入口：串行构建、记 RELOAD['last']、落盘，失败再把 traceback 打到 stderr
    （serve.err.log 因此带着崩溃现场，而不是一句异常名）。"""
    with _load_lock:
        ok, msg, tb = reload_zone(zone)
    RELOAD["last"][zone] = msg
    new = record_reload(zone, ok, msg, tb)
    if not ok and new:
        print(f"[kb_serve] RELOAD FAILED: {msg}\n{tb or ''}", file=sys.stderr, flush=True)
    return ok, msg

def _watch_snapshot():
    snap = {z: mtime(zone_index(z)) for z in ZONES}
    snap["alias"] = mtime(os.path.join(HERE, "alias_map.json"))
    return snap

def watcher():
    """每 10s 轮询索引与别名表 mtime，变更自动重载。失败不更新基线，下轮重试——
    chunks 已换 vec 未换的中间态会被行数断言挡住并保留旧档，直到三件齐了才切换。
    另每 LAG_EVERY_SEC 刷新一次语料/索引差集报警（C 组，只报不修）。"""
    base = _watch_snapshot()
    next_lag = 0.0
    while True:
        time.sleep(10)
        if not STATE["ready"]:
            base = _watch_snapshot()
            continue
        cur = _watch_snapshot()
        for z in ZONES:
            if cur[z] and cur[z] != base[z]:
                ok, msg = reload_and_record(z)
                if ok:
                    base[z] = cur[z]
                    print(f"[kb_serve] 自动重载 {msg}", flush=True)
        if time.time() >= next_lag:
            next_lag = time.time() + LAG_EVERY_SEC
            refresh_lag_alerts()
        if cur["alias"] != base["alias"]:
            f = STATE.get("main")
            if f is not None:
                f.doc._alias_map = None   # 别名表运行时读文件，清缓存即生效，无需重建索引
            base["alias"] = cur["alias"]
            RELOAD["last"]["alias"] = "alias_map 缓存已清"

def boot():
    try:
        with _load_lock:
            # 主库 provenance 模型不一致＝致命，不挂载并明说（防跨进程换档静默受害）
            fatal = check_provenance("main", HERE, "vec_m3.npy", MODEL_MAIN)
            if fatal:
                STATE["error"] = fatal
                print(f"[kb_serve] BOOT FAILED: {fatal}", flush=True)
                return
            f = _build_main(None)
            list(f.doc.embedder().embed(["预热"]))   # 真跑一次 m3：只构造对象不加载权重，首查仍会卡住
            _share_m3(f, f.doc._embedder)            # mem/epi/comp 共用同一个 m3 实例，省内存
            STATE["main"] = f
            for w in tail_prov_warnings():
                STATE["warn"].append(w); print(f"[kb_serve] {w}", flush=True)
            zh_dir = os.path.join(BASE, "kb_zh")
            if os.path.exists(os.path.join(zh_dir, "chunks.jsonl")):
                w = check_provenance("zh", zh_dir, "vec.npy", MODEL_ZH)
                if w:
                    STATE["warn"].append(w); print(f"[kb_serve] {w}", flush=True)
                else:
                    STATE["zh"] = _build_zh(None)   # e5 懒加载：首个专利查询才加载，不预热
            comp_dir = os.path.join(BASE, "kb_comp")
            if os.path.exists(os.path.join(comp_dir, "chunks.jsonl")):
                w = check_provenance("comp", comp_dir, "vec.npy", MODEL_MAIN)
                if w:
                    STATE["warn"].append(w); print(f"[kb_serve] {w}", flush=True)
                else:
                    STATE["comp"] = _build_comp(f.doc._embedder)
        STATE["ready"] = True
        refresh_lag_alerts()      # 重启后的第一件事之一就是看上次崩溃有没有把索引留在语料后面
        print("[kb_serve] READY", flush=True)
    except Exception as e:
        STATE["error"] = f"{type(e).__name__}: {e}"
        print(f"[kb_serve] BOOT FAILED: {STATE['error']}", flush=True)

DASHBOARD = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>海马体监控</title><meta http-equiv="refresh" content="5">
<style>
body{background:#111417;color:#d8dee4;font:14px/1.6 "Microsoft YaHei",monospace;margin:24px}
h1{font-size:18px}h2{font-size:14px;color:#8b949e;margin:18px 0 6px;border-bottom:1px solid #2d333b;padding-bottom:4px}
.lamp{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.g{background:#2ecc71;box-shadow:0 0 8px #2ecc71}.y{background:#f1c40f}.r{background:#e74c3c;box-shadow:0 0 8px #e74c3c}
.cards{display:flex;gap:12px;flex-wrap:wrap}.card{background:#161b22;border:1px solid #2d333b;border-radius:8px;padding:10px 16px;min-width:120px}
.card b{font-size:20px}.warn{background:#3a2a10;border:1px solid #b7791f;border-radius:8px;padding:8px 14px;margin:10px 0}
table{border-collapse:collapse;width:100%}td,th{padding:3px 10px;border-bottom:1px solid #21262d;text-align:left;font-size:13px}
th{color:#8b949e;font-weight:normal}.dim{color:#6e7681}.sup{text-decoration:line-through;color:#6e7681}
.tag{background:#1f6feb22;border:1px solid #1f6feb66;border-radius:4px;padding:0 6px;font-size:12px}
</style></head><body>
<h1><span id="lamp" class="lamp r"></span>海马体 · 8310</h1>
<div id="root">加载中…</div>
<script>
async function tick(){
  let s=null,dead=false;
  try{ s=await (await fetch('/stats',{signal:AbortSignal.timeout(4000)})).json(); }catch(e){ dead=true }
  const L=document.getElementById('lamp');
  if(dead||!s.ok){L.className='lamp '+(dead?'r':'y');
    document.getElementById('root').innerHTML=dead
      ?'<p>服务未响应。拉起命令：</p><pre class="dim">cd <仓库目录>/kb && python -X utf8 kb_serve.py</pre>'
      :'<p>'+(s.error?('启动失败：'+s.error):'模型加载中（约 1–2 分钟）…')+'</p>';return}
  L.className='lamp g';
  const up=Math.floor(s.uptime_s/60), h=Math.floor(up/60);
  const fmt=n=>n==null?'-':n;
  let html='<div class="cards">'
    +'<div class="card">主库文档块<br><b>'+s.zones.main_chunks+'</b></div>'
    +'<div class="card">记忆区块<br><b>'+s.zones.mem_chunks+'</b></div>'
    +'<div class="card">专利库块<br><b>'+s.zones.zh_chunks+'</b></div>'
    +'<div class="card">竞品库块<br><b>'+fmt(s.zones.comp_chunks)+'</b></div>'
    +'<div class="card">复盘区块<br><b>'+fmt(s.zones.epi_chunks)+'</b> <span class="dim">'+(s.zones.epi_chunks?'已转正':'未挂(无 PROMOTED)')+'</span></div>'
    +'<div class="card">运行<br><b>'+(h?h+'h ':'')+up%60+'m</b></div>'
    +'<div class="card">本进程查询<br><b>'+s.perf.n+'</b> <span class="dim">avg '+fmt(s.perf.avg_ms)+'ms · p95 '+fmt(s.perf.p95_ms)+'ms</span></div></div>';
  const ZN={main:'主库',mem:'记忆区',epi:'复盘区',zh:'专利库',comp:'竞品库'};
  const ZO=['main','mem','epi','zh','comp'];
  const esc=x=>String(x).replace(/</g,'&lt;');
  for(const z of (s.stale||[])){const c=s.corpus[z]||{};
    html+='<div class="warn">⚠ '+ZN[z]+' 语料比索引新（最新改动 '+c.latest_mtime+' '+esc(c.latest_file||'')+'）——需重建该区索引（服务会自动热重载，或 POST /reload）</div>'}
  for(const t of (s.lag||[])) html+='<div class="warn">⚠ '+esc(t)+'　（只报警：重建闸门是人走的）</div>';
  for(const z of ZO) if(s.disk&&s.disk.drift&&s.disk.drift[z])
    html+='<div class="warn">⚠ '+ZN[z]+' 内存态('+s.zones[z+'_chunks']+' 块) ≠ 磁盘态('+s.disk[z]+' 块)——热重载被行数/模型校验挡住时会一直不一致，先确认磁盘那份是过闸的档位（详见 /stats 的 reload 字段）</div>';
  html+='<h2>信息流 · 最近查询</h2><table><tr><th>时间</th><th>库</th><th>问题</th><th>Top 命中</th></tr>';
  for(const q of s.recent){const t=(q.ts||'').slice(5,16);
    html+='<tr><td class="dim">'+t+'</td><td>'+(q.kb||'main')+'</td><td>'+q.query.replace(/</g,'&lt;')+'</td><td class="dim">'+(q.hits||[]).slice(0,3).map(x=>x.split('/').pop()).join(' · ').replace(/</g,'&lt;')+'</td></tr>'}
  if(!s.recent.length)html+='<tr><td colspan="4" class="dim">暂无查询</td></tr>';
  html+='</table><h2>记忆区 · '+s.mem.total+' 条（superseded '+s.mem.superseded+' · 低置信 '+s.mem.low_conf+'）</h2><table><tr><th>条目</th><th>置信</th><th>创建</th><th>状态</th></tr>';
  for(const m of s.mem.entries){const sup=m.superseded_by&&m.superseded_by!='null';
    html+='<tr><td class="'+(sup?'sup':'')+'">'+m.file+'</td><td>'+(m.confidence||'-')+'</td><td class="dim">'+(m.created||'').slice(0,10)+'</td><td>'+(sup?'<span class="tag">被 '+m.superseded_by+' 取代 · 分数×0.3</span>':'活跃')+'</td></tr>'}
  html+='</table><h2>索引新鲜度（按区）</h2><table><tr><th>区</th><th>索引文件</th><th>索引更新</th><th>语料最新</th></tr>';
  for(const z of ZO){const f=s.index_files[z]||{},c=s.corpus[z]||{};
    html+='<tr><td>'+ZN[z]+'</td><td class="dim">'+esc(f.path||'-')+'</td><td>'+esc(f.mtime||'-')
      +'</td><td>'+(c.stale?'🟠 ':'')+esc(c.latest_mtime||'-')+' <span class="dim">'+esc(c.latest_file||'')+'</span>'+(c.stale?' <span class="tag">过期</span>':'')+'</td></tr>'}
  html+='</table><h2>重载记录 · 最近 10 次（落盘，重启后仍可查）</h2><table><tr><th>时间</th><th>区</th><th>结果</th><th>说明</th></tr>';
  for(const r of (s.reload_history||[]))
    html+='<tr><td class="dim">'+esc((r.ts||'').slice(5,16))+'</td><td>'+esc(r.zone)+'</td><td>'
      +(r.ok?'<span class="tag">ok</span>':'<span class="lamp r"></span>失败')+'</td><td class="dim">'
      +esc(r.message)+((r.traceback&&r.traceback.split('\\n').pop())?' ← '+esc(r.traceback.split('\\n').pop()):'')+'</td></tr>';
  if(!(s.reload_history||[]).length)html+='<tr><td colspan="4" class="dim">本次以来尚无重载</td></tr>';
  html+='<h2>命中计数 · '+((s.hits||{}).total||0)+' 次（本进程，仅展示）</h2><table><tr><th>区</th><th>文件:行</th><th>标题</th><th>次数</th></tr>';
  for(const h of ((s.hits||{}).top||[]))
    html+='<tr><td>'+ZN[h.zone]+'</td><td class="dim">'+esc(h.file)+':'+h.line+'</td><td>'+esc(h.heading)+'</td><td>'+h.count+'</td>';
  if(!((s.hits||{}).top||[]).length)html+='<tr><td colspan="4" class="dim">本进程尚无命中</td></tr>';
  if((s.hits||{}).note)html+='<tr><td colspan="4" class="dim">'+esc(s.hits.note)+'</td></tr>';
  document.getElementById('root').innerHTML=html+'</table>';
}
tick();setInterval(tick,5000);
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": STATE["ready"], "mem": bool(STATE["main"] and STATE["main"].has_mem),
                                    "zh": STATE["zh"] is not None, "comp": STATE.get("comp") is not None,
                                    "port": PORT,
                                    "error": STATE["error"] or None,
                                    "reload_fail": RELOAD["last_fail"],
                                    "lag": STATE["lag"],
                                    "warn": STATE["warn"] or None})
        if self.path == "/stats":
            return self._send(200, stats_json())
        if self.path in ("/", "/index.html"):
            body = DASHBOARD.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/search", "/reload"):
            return self._send(404, {"error": "use POST /search 或 POST /reload"})
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        spec = {} if (not raw and self.path == "/reload") else None   # /reload 允许空 body（默认 zone=all）
        for enc in ("utf-8", "gbk"):
            if spec is not None:
                break
            try:
                spec = json.loads(raw.decode(enc))
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        if spec is None:
            return self._send(400, {"error": "bad body: not utf-8/gbk json"})
        if self.path == "/reload":
            if not STATE["ready"]:
                return self._send(503, {"error": "loading, retry in a few seconds"})
            zone = str(spec.get("zone", "all"))
            ok, msg = reload_and_record(zone)
            return self._send(200 if ok else 500, {"ok": ok, "zone": zone, "message": msg})
        if not STATE["ready"]:
            if STATE["error"]:
                return self._send(503, {"error": f"boot failed: {STATE['error']}"})
            return self._send(503, {"error": "loading, retry in a few seconds"})
        q = str(spec.get("query", "")).strip()
        if not q:
            return self._send(400, {"error": "empty query"})
        kb = spec.get("kb", "main")
        if kb not in STATE or STATE[kb] is None:
            kb = "main"
        if spec.get("mode") not in MODES_OK:
            spec["mode"] = DEFAULT_MODE[kb]  # 缺省/非法档位一律落到该库验证过的配置
        spec["k"] = max(1, min(int(spec.get("k", 8)), 20))
        try:
            f = STATE[kb]
            t0 = time.time()
            # KB 实例在构造时已定格模型名，query 前缀随实例走，检索期间不再动 os.environ——
            # 双模型共进程也无需全局锁，请求之间没有共享可变状态。
            if not f.tails:      # 无尾部区的域（专利库、竞品域）只回文档块
                res = f.doc.search(q, k=spec["k"], mode=spec["mode"])
                out = [{"zone": "doc", "file": c["file"], "line": c["line"], "heading": c["heading"],
                        "score": round(float(s), 4), "text": c["text"]} for c, s in res]
            else:
                res = f.search(q, k=spec["k"], layout=spec.get("layout", "tiered"), mode=spec["mode"])
                out = [{"zone": z, "file": c["file"], "line": c["line"], "heading": c["heading"],
                        "score": round(float(s), 4), "text": c["text"][:1200]} for z, c, s in res]
            LATS.append(round((time.time() - t0) * 1000))
            # 命中计数只做面板展示，不回灌排序（加成须先过闸门）。专利/竞品域返回的
            # zone 字段恒为 "doc"，归档时换成实际库名，否则五区统计会全挤到 main。
            if not spec.get("replay"):
                zlabel = (lambda z: z) if f.tails else (lambda z: kb)
                with _hits_lock:
                    for r in out:
                        k = (zlabel(r["zone"]), r["file"], r["line"])
                        cur = HITS.get(k)
                        if cur is None:
                            HITS[k] = [1, r["heading"]]
                        else:
                            cur[0] += 1
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        if not spec.get("replay"):   # 回放不写日志、不计命中，否则回放本身污染下一轮回放的输入
            log_query({"ts": datetime.datetime.now().isoformat(timespec="seconds"),
                       "query": q, "kb": kb, "mode": spec["mode"],
                       "hits": [r["file"] for r in out[:5]]})
        self._send(200, out)

    def log_message(self, *a):
        pass

def _port_in_use(port):
    """主动 connect 探测端口是否已有监听者。
    Windows 上 HTTPServer 默认 allow_reuse_address=1 会让第二个实例 bind 同一端口**成功**
    （SO_REUSEADDR 在 Windows 语义不同），因此不能靠 bind 失败判双开，必须主动连一次。
    connect 只对活跃监听者成功，不受 TIME_WAIT 影响，重启不会误判。"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()

def main():
    # 先确认端口空闲再加载模型：已有实例时立即失败，不浪费 1–2 分钟模型加载、不吃双份内存
    if _port_in_use(PORT):
        print(f"[kb_serve] FATAL: 端口 {PORT} 已有服务在监听——已有实例在跑，本实例退出（不影响旧实例）。", flush=True)
        sys.exit(1)
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    except OSError as e:
        print(f"[kb_serve] FATAL: 端口 {PORT} 绑定失败（{e}），本实例退出。", flush=True)
        sys.exit(1)
    threading.Thread(target=boot, daemon=True).start()
    threading.Thread(target=watcher, daemon=True).start()
    print(f"[kb_serve] listening on 127.0.0.1:{PORT} (loading models…)", flush=True)
    srv.serve_forever()

if __name__ == "__main__":
    main()
