"""fuse.py — 文档区 + 记忆区融合检索（G 闸门的被测对象）。
用法: python fuse.py "查询" [-k 8] [--layout mixed|tiered] [--serve]
  tiered: 文档前 N 条、记忆后 M 条分层返回（默认，安全布局）
  mixed : RRF 混合排序，superseded 记忆分数 ×0.3（仅当闸门 PASS 后用）
"""
import argparse, json, os, re, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import jieba
jieba.setLogLevel(60)
from kb_search import KB, instr_for
import index as IX

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)

def load_kb(kbdir, model, vec_file):
    old = {k: os.environ.get(k) for k in ("KB_DIR", "KB_VEC", "KB_MODEL")}
    kbdir = os.path.abspath(kbdir)      # 相对目录会和 HERE 二次拼接，vec 档就找不到了
    vec_file = os.path.abspath(vec_file)
    os.environ["KB_DIR"] = kbdir
    os.environ["KB_VEC"] = vec_file
    os.environ["KB_MODEL"] = model
    import importlib
    import index as IX
    importlib.reload(IX)  # 让 HERE/CHUNKS 指向新目录
    import kb_search
    importlib.reload(kb_search)
    kb = kb_search.KB()
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    kb._kb_search = kb_search
    return kb, kb_search

def mem_flags(corpus_dir):
    """file -> (superseded?, confidence)"""
    out = {}
    if not os.path.isdir(corpus_dir):
        return out
    for fn in os.listdir(corpus_dir):
        if not fn.endswith(".md"):
            continue
        head = open(os.path.join(corpus_dir, fn), encoding="utf-8").read(2000)
        m = FM_RE.match(head)
        sup, conf = False, 1.0
        if m:
            for line in m.group(1).splitlines():
                if line.startswith("superseded_by:") and line.split(":", 1)[1].strip():
                    sup = True
                if line.startswith("confidence:"):
                    try:
                        conf = max(0.0, min(1.0, float(line.split(":", 1)[1].strip())))
                    except ValueError:
                        conf = 1.0
        out[fn] = (sup, conf)
    return out

class Fused:
    def __init__(self, model=os.environ.get("KB_MODEL", "bge-m3-onnx"),
                 doc_dir=None, mem_dir="", epi_dir=""):
        doc_dir = doc_dir or os.path.join(BASE, "kb")
        # mem_dir=None 表示该域压根没有尾部区（专利库/竞品域只有文档区）。
        # 不能写成 `mem_dir or 默认值`：None 会被静默当成"用默认"，域库于是挂上主库的
        # 记忆区与复盘区，用错的向量档（e5 查 m3 向量）返回一堆垃圾尾部命中。
        no_tails = mem_dir is None
        mem_dir = os.path.join(BASE, "kb", "mem") if mem_dir == "" else mem_dir
        epi_dir = os.path.join(BASE, "kb", "epi") if epi_dir == "" else epi_dir
        doc_vec = os.environ.get("KB_VENAME", "vec_m3.npy")
        self.doc, self._dmod = load_kb(doc_dir, model, os.path.join(doc_dir, doc_vec))
        # 尾部区（tiered 布局里只排在文档之后，永不与文档抢分）：mem 一直在，
        # epi 要 PROMOTED 标记存在才挂载——复盘区入库闸门是人走的，不是脚本自己走的。
        # mem_dir=None 表示这个域没有尾部（专利库就是：它只有文档区）。
        tails = [mem_dir] if mem_dir else []
        if not no_tails and os.path.exists(os.path.join(epi_dir, "PROMOTED")):
            tails.append(epi_dir)
        self.tails = []
        for d in tails:
            if os.path.exists(os.path.join(d, "chunks.jsonl")):
                kb, mod = load_kb(d, model, os.path.join(d, "vec.npy"))
                self.tails.append({"name": os.path.basename(d), "kb": kb, "mod": mod,
                                   "flags": mem_flags(os.path.join(d, "corpus"))})
        self.model = model
        self.has_mem = any(t["name"] == "mem" for t in self.tails)

    def _decay(self, tail, c):
        sup, conf = tail["flags"].get(c["file"].split("/")[-1], (False, 1.0))
        return (0.3 if sup else 1.0), conf

    def search(self, query, k=8, layout="tiered", mode="hybrid-v2"):
        dk = k if layout == "tiered" else k + 4
        vec_modes = ("vec", "hybrid", "hybrid-v2", "vec-r", "hybrid-r")
        qv = self.doc.query_vector(query) if self.tails and mode in vec_modes else None
        docs = self.doc.search(query, k=dk, mode=mode, qv=qv)
        if not self.tails:
            return [("doc", c, s) for c, s in docs]
        tail_hits = []
        for t in self.tails:
            tail_hits.append((t, [(c, s * self._decay(t, c)[0])
                                  for c, s in t["kb"].search(query, k=min(4, k), mode=mode, qv=qv)]))
        if layout == "tiered":
            out = [("doc", c, s) for c, s in docs]
            for t, hits in tail_hits:
                out += [(t["name"], c, s) for c, s in hits]
            return out
        ranks = {}
        # 名次必须按"各自列表内的排名"算（RRF 口径）。用池内位置会把尾部整体压到
        # 文档之后，mixed 就退化成 tiered，测出来"不涨"其实是被这么埋掉的。
        pool = [("doc", c, s, None, rank) for rank, (c, s) in enumerate(docs)]
        for t, hits in tail_hits:
            pool += [(t["name"], c, s, t, rank) for rank, (c, s) in enumerate(hits)]
        for name, c, s, t, rank in pool:
            # 衰减必须真正影响排序：旧实现按 1/(60+rank) 单变量排序，×0.3 与 confidence
            # 只进 score 不进顺序，superseded 记忆照样排到前面
            sup, conf = self._decay(t, c) if t is not None else (1.0, 1.0)
            weight = (1 / (60 + rank)) * sup * conf
            key = (name, c["id"])
            if weight > ranks.get(key, (0,))[0]:
                ranks[key] = (weight, name, c, s)
        top = sorted(ranks.values(), key=lambda x: -x[0])[:k]
        return [(o, c, s) for _, o, c, s in top]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--layout", default="tiered", choices=["tiered", "mixed"])
    ap.add_argument("--mode", default="hybrid-v2")
    ap.add_argument("--serve", action="store_true")
    a = ap.parse_args()
    f = Fused()
    if a.serve:
        print(json.dumps({"ready": True, "mem_indexed": f.has_mem}), flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line: continue
            spec = json.loads(line) if line.startswith("{") else {"query": line}
            res = f.search(spec.get("query", ""), spec.get("k", a.k), spec.get("layout", a.layout), spec.get("mode", a.mode))
            print(json.dumps([{"zone": z, "file": c["file"], "line": c["line"], "heading": c["heading"],
                               "score": round(float(s), 4), "text": c["text"][:600]} for z, c, s in res],
                             ensure_ascii=False), flush=True)
            import datetime
            with open(os.path.join(BASE, "kb", "query_log.jsonl"), "a", encoding="utf-8") as w:
                w.write(json.dumps({"ts": datetime.datetime.now().isoformat(timespec="seconds"),
                                    "query": spec.get("query", ""), "mode": spec.get("mode", a.mode),
                                    "hits": [c["file"] for z, c, s in res[:5]]}, ensure_ascii=False) + "\n")
        return
    if not a.query: ap.error("需要 query 或 --serve")
    res = f.search(a.query, a.k, a.layout, a.mode)
    for i, (z, c, s) in enumerate(res, 1):
        print(f"{i:2d} [{z}] [{s:.3f}] {c['file']} L{c['line']} #{c['heading'][:40]}\n    {c['text'][:110]}")
    import datetime
    with open(os.path.join(BASE, "kb", "query_log.jsonl"), "a", encoding="utf-8") as w:
        w.write(json.dumps({"ts": datetime.datetime.now().isoformat(timespec="seconds"),
                            "query": a.query, "mode": a.mode,
                            "hits": [c["file"] for z, c, s in res[:5]]}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
