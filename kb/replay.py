"""replay.py — 用真实查询日志回放检索，落快照并与上一份对比。

改 chunker/embedding/融合档位之前先跑一份快照，改完再跑一份并 diff：
金标准 55 题只看"该不该命中"，回放看的是"平时真的在查的东西排名动没动"——
两批索引在 eval 上分数一样时，回放能暴露长尾查询的静默漂移。

用法:
  python replay.py --dry-run                 # 只看去重后的查询集，不查服务
  python replay.py                           # 回放 → kb/replay_YYYYmmdd-HHMM.jsonl
  python replay.py --diff latest             # 回放并和上一份快照比
  python replay.py --diff kb/replay_x.jsonl  # 和指定快照比
  python replay.py --since 2026-11-01 --top 80 --min-count 2

回放请求带 "replay": true，服务端不写 query_log、不计命中计数。
"""
import argparse, datetime, glob, json, os, re, sys, time, urllib.request

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "query_log.jsonl")
URL = os.environ.get("KB_URL", "http://127.0.0.1:8310")
TOPK = 5


def norm(q):
    """去重键：空白与全半角标点差异不算不同查询。"""
    return re.sub(r"[\s,，。;；:：?？!！、]+", "", q).lower()


def load_queries(since=None, top=None, min_count=1):
    """按 (kb, 归一化查询) 去重：保留最近一次的原样写法，count 是出现次数。"""
    best, cnt = {}, {}
    try:
        lines = open(LOG, encoding="utf-8").read().splitlines()
    except OSError:
        sys.exit(f"没有查询日志：{LOG}（先让 8310 跑几天真实查询）")
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        q = str(e.get("query", "")).strip()
        kb = e.get("kb") or "main"
        if len(norm(q)) < 2:
            continue
        ts = e.get("ts", "")
        if since and ts[:10] < since:
            continue
        key = (kb, norm(q))
        cnt[key] = cnt.get(key, 0) + 1
        if key not in best or ts >= best[key]["ts"]:
            best[key] = {"query": q, "kb": kb, "mode": e.get("mode") or "", "ts": ts}
    out = []
    for key, e in best.items():
        e["count"] = cnt[key]
        if e["count"] >= min_count:
            out.append(e)
    out.sort(key=lambda e: (-e["count"], e["ts"]))
    return out[:top] if top else out


def call(path, body=None):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(URL + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def replay(queries):
    try:
        h = call("/health")
    except OSError as e:
        sys.exit(f"8310 不可用（{e}）；先拉起 kb_serve.py")
    if not h.get("ok"):
        sys.exit(f"8310 未就绪：{h}")
    snaps, lats, miss = [], [], []
    for i, e in enumerate(queries, 1):
        body = {"query": e["query"], "k": 8, "kb": e["kb"], "replay": True}
        t0 = time.time()
        try:
            out = call("/search", body)
        except Exception as ex:                       # 单条失败不应拖垮整批快照
            miss.append((e["query"], f"{type(ex).__name__}: {ex}"))
            continue
        ms = round((time.time() - t0) * 1000)
        lats.append(ms)
        snaps.append({"query": e["query"], "kb": e["kb"], "n_log": e["count"], "ms": ms,
                      "hits": [{"zone": r["zone"], "file": r["file"], "line": r["line"],
                                "score": r["score"]} for r in out[:TOPK]]})
        if i % 10 == 0 or i == len(queries):
            print(f"  {i}/{len(queries)} …", file=sys.stderr, flush=True)
    lats.sort()
    print(f"回放 {len(snaps)} 条  延迟 p50={lats[len(lats)//2] if lats else '-'}ms "
          f"p95={lats[int(len(lats)*0.95)-1] if lats else '-'}ms  失败 {len(miss)}")
    for q, why in miss[:10]:
        print(f"  FAIL {q}  {why}")
    return snaps


def key(h):
    return (h["zone"], h["file"], h["line"])


def side(keys):
    return ", ".join(f"{f}:{l}" for _, f, l in sorted(keys)) or "-"


def diff(old, new):
    om = {(e["kb"], e["query"]): e for e in old}
    nm = {(e["kb"], e["query"]): e for e in new}
    gone = [k for k in om if k not in nm]
    added = [k for k in nm if k not in om]
    changed, same, overlaps = [], 0, []
    for k in nm:
        if k not in om:
            continue
        a = {key(h) for h in om[k]["hits"]}
        b = {key(h) for h in nm[k]["hits"]}
        if not a and not b:
            continue
        ov = len(a & b) / max(len(a | b), 1)
        overlaps.append(ov)
        if a != b:
            changed.append((k, a - b, b - a, ov))
    n = len(changed) + same
    print(f"\n对比：共有 {len(overlaps)} 条（重叠率均值 {sum(overlaps)/len(overlaps):.0%}）"
          f" · 名次集合有变 {len(changed)} 条 · 仅旧快照 {len(gone)} 条 · 仅新快照 {len(added)} 条")
    for (kb, q), out, inn, ov in sorted(changed, key=lambda x: x[3]):
        print(f"\n[{kb}] {q}  overlap@5={ov:.0%}")
        if out:
            print(f"   - 掉出: {side(out)}")
        if inn:
            print(f"   + 进入: {side(inn)}")
    if not changed:
        print("Top5 集合完全一致（分数细节请看两份快照）")
    score_moves = []
    for k in nm:
        if k in om and om[k]["hits"] and nm[k]["hits"]:
            a, b = om[k]["hits"][0], nm[k]["hits"][0]
            if key(a) == key(b):
                score_moves.append(abs(b["score"] - a["score"]))
    if score_moves:
        print(f"同一条 top1 的分数漂移：均值 {sum(score_moves)/len(score_moves):.3f} "
              f"最大 {max(score_moves):.3f}（n={len(score_moves)}）")


def latest_snapshot(exclude=None):
    fs = [p for p in glob.glob(os.path.join(HERE, "replay_*.jsonl")) if p != exclude]
    return max(fs, key=os.path.getmtime) if fs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="只回放该日期(YYYY-MM-DD)之后的查询")
    ap.add_argument("--top", type=int, default=60, help="最多回放多少条（按出现次数降序）")
    ap.add_argument("--min-count", type=int, default=1, help="只看出现过 N 次以上的查询")
    ap.add_argument("--dry-run", action="store_true", help="只列查询集，不调服务")
    ap.add_argument("--out", help="快照输出路径")
    ap.add_argument("--diff", help="对比快照：latest 或文件路径")
    a = ap.parse_args()

    qs = load_queries(a.since, a.top, a.min_count)
    if not qs:
        sys.exit("没有符合条件的查询")
    print(f"去重后 {len(qs)} 条查询（按出现次数降序；kb 分布 "
          f"{ {k: sum(1 for e in qs if e['kb'] == k) for k in sorted({e['kb'] for e in qs})} }）")
    if a.dry_run:
        for e in qs[:30]:
            print(f"  [{e['kb']:4s}] ×{e['count']} {e['query']}")
        if len(qs) > 30:
            print(f"  … 另 {len(qs) - 30} 条")
        return

    snaps = replay(qs)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")   # 分钟级会覆盖上一份，diff 就失去基线
    out = a.out or os.path.join(HERE, f"replay_{stamp}.jsonl")
    with open(out, "w", encoding="utf-8") as w:
        for s in snaps:
            w.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("快照:", os.path.basename(out))
    if a.diff:
        prev = latest_snapshot(exclude=out) if a.diff == "latest" else a.diff
        if not prev or not os.path.exists(prev):
            print("没有可对比的旧快照（首次回放即基线）")
            return
        print("对比基线:", os.path.basename(prev))
        diff([json.loads(l) for l in open(prev, encoding="utf-8") if l.strip()], snaps)


if __name__ == "__main__":
    main()
