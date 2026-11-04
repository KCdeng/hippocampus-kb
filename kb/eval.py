"""eval.py — G0 评测：grep基线 vs BM25 vs 向量 vs 混合。指标 hit@1/3/5、MRR。"""
import datetime, json, os, re, sys
import numpy as np
import jieba
sys.stdout.reconfigure(encoding="utf-8")
from index import tokenize, HERE
from kb_search import KB, QUERY_INSTR

def norm(s):
    return re.sub(r"\s+", "", s)

def same_file(a, b):
    """同一文件判定：允许一方是另一方的相对子路径（md/x.md vs x.md），但必须落在目录边界上。
    裸 endswith 会把 myfoo.md 判成 foo.md——文件名后缀相同就被算命中，分数会虚高。"""
    a, b = norm(a).lower().replace("\\", "/"), norm(b).lower().replace("\\", "/")
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)

def load_gold():
    ev = json.load(open(os.path.join(HERE, "eval_set.json"), encoding="utf-8"))
    filecache = {}
    for e in ev:
        p = os.path.join(os.environ.get("KB_ROOT", os.path.join(os.path.dirname(HERE), "corpus")), e["gold_file"].replace("/", os.sep))
        if p not in filecache:
            filecache[p] = norm(open(p, encoding="utf-8", errors="replace").read()) if os.path.exists(p) else ""
        e["_file_norm"] = filecache[p]
    missing = [e["qid"] for e in ev if not e["_file_norm"]]
    if missing:
        # gold 文件搬家/删除会让该题永远判负，看起来像检索退化——必须显式失败
        raise SystemExit(f"eval_set 中 gold_file 不存在: {missing}")
    return ev

def grep_baseline(kb, query):
    """模拟智能体 grep：查询切词后按子串出现次数给块打分（关键词精确匹配）。"""
    terms = [t for t in re.split(r"[\s,，。;；:：]+", query) if len(t) > 1]
    terms += [t for t in set(tokenize(query)) if len(t) > 2][:8]
    scores = []
    for c in kb.chunks:
        n = sum(norm(c["text"]).count(norm(t)) + 0.5 * norm(c["heading"]).count(norm(t)) for t in terms)
        scores.append(n)
    scores = np.array(scores)
    return [int(i) for i in np.argsort(-scores)]

def evaluate(topk=10):
    # 档位必须由评测脚本自己钉死：不设 KB_VEC/KB_MODEL 时 kb_search 会静默退到
    # bge-small(512d)+vec.npy 陈旧件，测出来的数字与生产无关。
    os.environ.setdefault("KB_VEC", os.path.join(HERE, "vec_m3.npy"))
    os.environ.setdefault("KB_MODEL", "bge-m3-onnx")
    kb = KB()
    ev = load_gold()
    methods = {}
    methods["grep"] = lambda q: [i for i in _order(grep_baseline(kb, q))]
    methods["bm25"] = lambda q: kb_rank(kb, q, "bm25")
    methods["vec"] = lambda q: kb_rank(kb, q, "vec")
    methods["hybrid"] = lambda q: kb_rank(kb, q, "hybrid")
    methods["hybrid-v2"] = lambda q: kb_rank(kb, q, "hybrid-v2")
    methods["bm25-r"] = lambda q: kb_rank(kb, q, "bm25-r")
    methods["hybrid-r"] = lambda q: kb_rank(kb, q, "hybrid-r")
    if len(sys.argv) > 1:
        sel = sys.argv[1].split(",")
        for m in list(methods):
            if m not in sel:
                del methods[m]

    def _order(lst): return lst

    results = {m: {"hit1": 0, "hit3": 0, "hit5": 0, "filehit5": 0, "rr": 0.0, "n": 0} for m in methods}
    perq = []
    for e in ev:
        q = e["question"]
        row = {"qid": e["qid"], "type": e.get("type", "?")}
        for m, fn in methods.items():
            idxs = fn(q)[:topk]
            r = results[m]; r["n"] += 1
            strict_rank, file_rank = None, None
            for rank, i in enumerate(idxs):
                c = kb.chunks[i]
                is_same = same_file(c["file"], e["gold_file"])
                strict = is_same and norm(e["gold_anchor"]) in norm(c["text"])
                if strict and strict_rank is None:
                    strict_rank = rank
                if is_same and file_rank is None:
                    file_rank = rank
            if strict_rank is not None:
                r["rr"] += 1 / (strict_rank + 1)
                r["hit1"] += strict_rank < 1
                r["hit3"] += strict_rank < 3
                r["hit5"] += strict_rank < 5
            r["filehit5"] += file_rank is not None and file_rank < 5
            row[m] = strict_rank + 1 if strict_rank is not None else None
        perq.append(row)

    print(f"{'method':10s} {'hit@1':>7s} {'hit@3':>7s} {'hit@5':>7s} {'MRR':>7s} {'文件级@5':>9s}")
    for m, r in results.items():
        n = r["n"]
        print(f"{m:10s} {r['hit1']/n:7.1%} {r['hit3']/n:7.1%} {r['hit5']/n:7.1%} "
              f"{r['rr']/n:7.3f} {r['filehit5']/n:9.1%}")
    # 按题型拆分
    for ty in ("keyword", "semantic", "multi"):
        sub = [p for p in perq if p["type"] == ty]
        if not sub: continue
        line = f"  [{ty} n={len(sub)}] "
        for m in methods:
            hits = sum(1 for p in sub if p[m] and p[m] <= 5)
            line += f"{m}:{hits/len(sub):.0%} "
        print(line)
    primary = next((m for m in ("hybrid-v2", "hybrid") if m in methods), list(methods)[-1])
    miss = [p["qid"] for p in perq if not (p[primary] and p[primary] <= 5)]
    print(f"{primary} 未命中题:", miss if miss else "无")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    path = os.path.join(HERE, f"eval_per_query_{stamp}.json")
    json.dump(perq, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("per-query 明细:", os.path.basename(path))
    verdict(primary, results[primary], len(kb.chunks))

def verdict(primary, r, n_chunks):
    """闸门判据来自 baseline.json，不靠记忆里的数字。"""
    try:
        base = json.load(open(os.path.join(HERE, "baseline.json"), encoding="utf-8"))
    except OSError:
        print("无 baseline.json，跳过闸门判定"); return
    n = r["n"] or 1
    hit5, file5 = r["hit5"] / n, r["filehit5"] / n
    g = base["gate"]
    if n_chunks != base["baseline"]["chunks"]:
        print(f"⚠ 块数 {n_chunks} ≠ 基线 {base['baseline']['chunks']}：这是另一代索引，"
              f"分数不可与基线直接比（且必须同步重嵌 vec）")
    ok = hit5 >= g["hit5_min"] and file5 >= g["filehit5_min"]
    print(f"闸门({base['baseline']['tag']} hit@5≥{g['hit5_min']:.1%} 且 文件级@5≥{g['filehit5_min']:.1%}) "
          f"→ {'PASS' if ok else 'FAIL'}：{primary} hit@5 {hit5:.1%} / 文件级@5 {file5:.1%}")
    if not ok:
        sys.exit(1)

def kb_rank(kb, query, mode):
    ids = {c["id"]: i for i, c in enumerate(kb.chunks)}
    return [ids[c["id"]] for c, s in kb.search(query, k=10, mode=mode)]

if __name__ == "__main__":
    evaluate()
