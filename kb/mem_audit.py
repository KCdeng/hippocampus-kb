"""mem_audit.py — 记忆区审计（A2 冲突候选）与整合（A3 合并取代）。

只做"可疑"判定，不做语义裁决：向量/标题只告诉你哪几条在讲同一件事，
两句话是不是真的互相矛盾，得人看。合并也只在显式 `merge --apply` 时发生，
默认 dry-run；confidence<0.5 的旧档照样不参与自动合并。

相似度用**去中心化余弦**：bge-m3 对同域中文的裸余弦全堆在 0.95 上下（各向异性），
按 0.90 取阈值会把整个记忆区并成一个簇；减掉语料均值后同项目对 ~0.8、跨主题 ~0.1。

用法:
  python mem_audit.py                      # 审计：完整性 + 相似簇
  python mem_audit.py --threshold 0.93     # 收紧阈值
  python mem_audit.py --json               # 机器可读
  python mem_audit.py merge mem-0003 mem-0011 --title "…" --body-file merged.md --apply
"""
import argparse, glob, json, os, re, sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research/hippocampus
MEM = os.environ.get("KB_MEM_DIR") or os.path.join(BASE, "kb", "mem")  # 与 kb_remember 同源
CORPUS = os.path.join(MEM, "corpus")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kb_remember as KR


# ---------- 读取 ----------

def load_entries():
    out = []
    for p in sorted(glob.glob(os.path.join(CORPUS, "mem-*.md"))):
        fm = KR.parse_fm(p)
        raw = open(p, encoding="utf-8").read()
        title = raw.split("# ", 1)[1].split("\n", 1)[0].strip() if "# " in raw else ""
        try:
            conf = float(fm.get("confidence") or 1.0)
        except ValueError:
            conf = 1.0
        out.append({"id": fm.get("id") or os.path.basename(p)[:-3],
                    "file": os.path.basename(p),
                    "title": title, "confidence": conf,
                    "superseded_by": (fm.get("superseded_by") or "").strip(),
                    "created": fm.get("created", ""),
                    "sources": fm.get("sources", "[]")})
    return out


def vec_rows():
    """file -> [vec 行号]。mem/vec.npy 与 chunks.jsonl 行对齐（kb_remember 校验过）。"""
    chunks = [json.loads(l) for l in open(os.path.join(MEM, "chunks.jsonl"), encoding="utf-8") if l.strip()]
    vec = np.load(os.path.join(MEM, "vec.npy")).astype(np.float32)
    if vec.shape[0] != len(chunks):
        raise SystemExit(f"mem 索引不同代：vec={vec.shape[0]} chunks={len(chunks)}，先重建 mem")
    vec /= (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9)
    rows = {}
    for i, c in enumerate(chunks):
        rows.setdefault(os.path.basename(c["file"]), []).append(i)
    return rows, vec


def file_sim(rows, vec, a, b):
    """两条记忆的相似度 = 各自块之间余弦的最大值（多块条目取最相关的那一组）。"""
    if a not in rows or b not in rows:
        return 0.0
    return float(np.max(vec[rows[a]][:, None, :] @ vec[rows[b]][None, :, :].transpose(0, 2, 1)))


def title_sim(a, b):
    """字符二元组 Jaccard。中文短标题用词袋太重，二元组对"修订版/续"这类后缀不敏感。"""
    def bg(s):
        s = re.sub(r"[\s（(].*?[）)]|修订版|补充|更新", "", s)
        return {s[i:i + 2] for i in range(len(s) - 1)} or {s}
    A, B = bg(a), bg(b)
    return len(A & B) / max(len(A | B), 1)


def bigrams(s):
    return {s[i:i + 2] for i in range(len(s) - 1)}


def body_of(fn):
    raw = open(os.path.join(CORPUS, fn), encoding="utf-8").read()
    raw = re.sub(r"\A---\n.*?\n---\n", "", raw, flags=re.S)
    return re.sub(r"[\s#*|>\-]+", "", raw)


# 只在标题里认"纠错口径"：正文到处是"作废/不是"，按正文判会把同项目的台账全标成冲突。
NEG_TITLE = ("不是", "而非", "应改为", "纠正", "更正", "修正", "推翻", "不成立", "误读")
STOP = {"的", "和", "与", "及", "或", "为", "在", "用", "按", "并", "是", "了", "对", "有", "无"}


def title_tokens(s):
    import jieba
    jieba.setLogLevel(60)
    return {t for t in jieba.lcut(s) if len(t) >= 2 and t not in STOP}


def centered(rows, vec):
    """去中心化：bge-m3 对各条记忆的裸余弦全在 0.95 上下（各向异性），
    减掉语料均值后同项目对 ~0.8、跨主题对 ~0.1，绝对阈值才有意义。"""
    c = vec - vec.mean(axis=0)
    return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)


# ---------- 审计 ----------

def integrity(entries):
    ids = {e["id"] for e in entries}
    probs = []
    for e in entries:
        sup = e["superseded_by"]
        if sup and sup not in ids:
            probs.append(f"{e['id']} 的 superseded_by={sup} 指向不存在的条目")
        if e["confidence"] < 0.5:
            probs.append(f"{e['id']} confidence={e['confidence']} < 0.5（规程要求不得入库，属历史遗留）")
        if not e["title"]:
            probs.append(f"{e['id']} 缺标题")
    byid = {e["id"]: e for e in entries}
    for e in entries:                      # 环检测：沿 superseded 链最多走 n 步
        seen, cur = [], e["id"]
        for _ in range(len(entries) + 1):
            nxt = byid.get(cur, {}).get("superseded_by", "")
            if not nxt:
                break
            if nxt in seen or nxt == e["id"]:
                probs.append(f"superseded 链成环：{e['id']} → … → {cur} → {nxt}")
                break
            seen.append(cur)
            cur = nxt
    return probs


def candidates(active, rows, vec, thr, cthr):
    """人工裁决候选。两条规则（任一命中即列出，不做自动合并）：
    dup  = 去中心化余弦 ≥ thr 且（标题二元组 Jaccard ≥ 0.25 或 正文二元组包含率 ≥ 0.30）
           —— 同题/同事实，通常是"同一条知识写了两次"。
    conf = 去中心化余弦 ≥ cthr 且其中一条标题带否决词（不是/作废/修正…）且它更晚
           —— 疑似纠错没走 correct，旧结论还挂着活跃标志。
    单纯"同主题"（项目内多条事实）不算候选，否则每个项目都会被判成一大簇。"""
    c = centered(rows, vec)
    bodies = {e["id"]: body_of(e["file"]) for e in active}
    grams = {k: bigrams(v) for k, v in bodies.items()}
    order = {e["id"]: i for i, e in enumerate(active)}
    dups, conf = [], []
    for i, a in enumerate(active):
        for b in active[i + 1:]:
            cos = file_sim(rows, c, a["file"], b["file"])
            ga, gb = grams[a["id"]], grams[b["id"]]
            cont = len(ga & gb) / max(min(len(ga), len(gb)), 1) if ga and gb else 0.0
            p = {"a": a["id"], "b": b["id"], "cos": round(cos, 3),
                 "title_j": round(title_sim(a["title"], b["title"]), 2), "cont": round(cont, 2)}
            # 短条目（一两句话）的向量会被句式拉偏，但标题几乎相同本身就是强证据
            if p["title_j"] >= 0.85:
                dups.append(dict(p, reason="标题几乎相同"))
                continue
            if cos >= thr and (p["title_j"] >= 0.25 or p["cont"] >= 0.30):
                dups.append(dict(p, reason="同题/同事实"))
                continue
            later, older = (b, a) if order[b["id"]] > order[a["id"]] else (a, b)
            shared = title_tokens(later["title"]) & title_tokens(older["title"])
            # 否决词必须作用在共同主题上：mem-0022 标题里的"用户四次推翻我的建议"不是纠正
            # 别的记忆，拿窗口判邻接就不会把它算成冲突。
            neg_err = ""
            for w in NEG_TITLE:
                pos = later["title"].find(w)
                if pos < 0:
                    continue
                win = later["title"][max(0, pos - 2):pos + len(w) + 5]   # 否决词得紧贴被否定的对象
                if any(t in win for t in shared):
                    neg_err = w
                    break
            if cos >= cthr and len(shared) >= 2 and neg_err:
                conf.append(dict(p, shared=sorted(shared), neg=neg_err,
                                 reason=f"{later['id']} 标题以「{neg_err}」否定共有主题，旧的没被 supersede"))
    return dups, conf, order


def group_pairs(pairs):
    """把候选对按连通关系收成合并建议簇。"""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        parent[find(p["a"])] = find(p["b"])
    g = {}
    for p in pairs:
        g.setdefault(find(p["a"]), set()).update((p["a"], p["b"]))
    return [sorted(v) for v in g.values() if len(v) > 1]


def audit(args):
    entries = load_entries()
    active = [e for e in entries if not e["superseded_by"]]
    probs = integrity(entries)
    try:
        rows, vec = vec_rows()
        dups, conf, _ = candidates(active, rows, vec, args.threshold, args.conflict_threshold)
    except (OSError, SystemExit) as e:
        print(f"跳过候选检测：{e}")
        dups, conf = [], []
    byid = {e["id"]: e for e in entries}

    if args.json:
        print(json.dumps({"total": len(entries), "active": len(active), "problems": probs,
                          "duplicates": dups, "conflicts": conf,
                          "merge_groups": group_pairs(dups), "entries": entries},
                         ensure_ascii=False, indent=1))
        return 0

    def show(pairs):
        for p in pairs:
            extra = f" 共有词={'/'.join(p['shared'])}" if p.get("shared") else ""
            print(f"\n  {p['a']} ~ {p['b']}  cos={p['cos']} 标题J={p['title_j']} 正文包含={p['cont']}{extra}")
            for mid in (p["a"], p["b"]):
                e = byid[mid]
                print(f"    {mid} conf={e['confidence']} created={e['created'][:10]} {e['title'][:60]}")

    print(f"记忆区：{len(entries)} 条（活跃 {len(active)} · 已被取代 {len(entries) - len(active)}）")
    print(f"完整性：{'无问题' if not probs else str(len(probs)) + ' 项'}")
    for p in probs:
        print("  ✗", p)
    print(f"\n疑似重复/同事实（cos≥{args.threshold} 且 标题或正文重合）：{len(dups)} 对")
    show(dups)
    print(f"\n疑似修订未取代（cos≥{args.conflict_threshold} 且 较新一条标题带否决词）：{len(conf)} 对")
    show(conf)
    for g in group_pairs(dups):
        print(f"\n  合并建议: python kb/mem_audit.py merge {' '.join(g)} --title \"…\" --body-file merged.md --apply")
    for p in conf:
        a, b = byid[p["a"]], byid[p["b"]]
        old, new = (a, b) if a["created"] <= b["created"] else (b, a)
        print(f"  取代建议: python kb/kb_remember.py correct --id {old['id']} --body \"…（写清新口径）\""
              f"   # 若 {new['id']} 已覆盖旧结论")
    if not dups and not conf:
        print("\n未发现需要人工裁决的重复或冲突。")
    print("\n说明：同主题但讲不同事实的条目属正常，不列。矛盾与否需人判断——本工具只给证据。")
    return 1 if probs else 0


# ---------- 整合 ----------

def merge(args):
    entries = {e["id"]: e for e in load_entries()}
    ids = args.ids
    if len(ids) < 2:
        sys.exit("merge 至少给两条 id")
    for mid in ids:
        if mid not in entries:
            sys.exit(f"找不到 {mid}")
        if entries[mid]["superseded_by"]:
            sys.exit(f"{mid} 已被 {entries[mid]['superseded_by']} 取代，不能再合并"
                     f"（先合并到最新那条，再取代它）")
    body = args.body or (open(args.body_file, encoding="utf-8").read() if args.body_file else None)
    if not body:
        sys.exit("需要 --body 或 --body-file（合并后的正文由人写，工具不代替你下结论）")
    old = [entries[m] for m in ids]
    conf = args.confidence if args.confidence is not None else min(e["confidence"] for e in old)
    if conf < 0.5:
        sys.exit(f"合并后 confidence={conf} < 0.5，不入库")
    srcs = sorted({s for e in old for s in json.loads(e["sources"] or "[]")})
    print(f"将写入：{args.title}（conf={conf}，取代 {', '.join(ids)}，来源 {len(srcs)} 个）")
    if not args.apply:
        print("dry-run：确认无误后加 --apply 执行")
        return

    # 写之前先把所有旧文件读一遍并校验，避免写到一半才发现某条不能改
    raws = []
    for m in ids:
        raw = open(os.path.join(CORPUS, f"{m}.md"), encoding="utf-8").read()
        if "superseded_by:" not in raw:
            sys.exit(f"{m} 的 frontmatter 缺 superseded_by:，拒绝改写")
        if re.search(r"(?m)^superseded_by:[ \t]*\S", raw):   # \s 会吃换行，误把下一行当值
            sys.exit(f"{m} 的 superseded_by 已有值，拒绝覆盖")
        raws.append((m, raw))

    mid, lock = KR.next_id()
    try:
        path = KR.write_md(mid, args.title, body, conf, srcs, args.task or "A3-合并")
        for m, raw in raws:                # 先落新条目，再打指针：中途失败最多留个孤立新条目，不会造成悬空指针
            open(os.path.join(CORPUS, f"{m}.md"), "w", encoding="utf-8").write(
                re.sub(r"(?m)^superseded_by:", f"superseded_by: {mid}", raw, count=1))
            print(f"superseded {m} -> {mid}")
    finally:
        KR._release(lock)
    KR.rebuild_mem()
    print("written:", os.path.relpath(path, BASE))
    print("提示：8310 须重启才能看到合并后的 mem 索引。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.90, help="重复候选的去中心化余弦阈值")
    ap.add_argument("--conflict-threshold", type=float, default=0.70, help="修订候选的余弦阈值")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    m = sub.add_parser("merge")
    m.add_argument("ids", nargs="+")
    m.add_argument("--title", required=True)
    m.add_argument("--body"); m.add_argument("--body-file")
    m.add_argument("--confidence", type=float)
    m.add_argument("--task", default="")
    m.add_argument("--apply", action="store_true", help="真的写入（默认 dry-run）")
    a = ap.parse_args()
    sys.exit(merge(a) if a.cmd == "merge" else audit(a))


if __name__ == "__main__":
    main()
