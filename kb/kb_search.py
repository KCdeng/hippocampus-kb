"""kb_search.py — 混合检索 CLI。用法: python kb_search.py "查询" [-k 8] [--mode hybrid|bm25|vec]"""
import argparse, json, os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import jieba
from index import tokenize, CHUNKS, HERE

QUERY_INSTR = "为这个句子生成表示以用于检索相关文章："
PASSAGE_INSTR = ""
_MODEL_CFG = {
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "jinaai/jina-embeddings-v2-base-zh": ("", ""),
    "bge-m3-onnx": ("", ""),
}

def instr_for(model):
    for key, (qi, pi) in _MODEL_CFG.items():
        if key in model:
            return qi, pi
    return QUERY_INSTR, PASSAGE_INSTR

class KB:
    @staticmethod
    def _default_vec():
        """KB_VEC 未设时的缺省档：优先生产 vec_m3.npy，退回 vec.npy（专利库就是这个名字）。
        历史上默认直接指向 vec.npy，主库跑一次就把 bge-small 陈旧件当成向量档用了。"""
        for name in ("vec_m3.npy", "vec.npy"):
            p = os.path.join(HERE, name)
            if os.path.exists(p):
                return p
        return os.path.join(HERE, "vec.npy")

    def __init__(self, embedder=None):
        # 模型名在构造时定格：load_kb 会临时设好 KB_MODEL 再建实例。之后按请求读 env
        # 会让双模型共进程的常驻服务不得不逐请求切环境并加全局锁（检索被串行化）。
        self.model = os.environ.get("KB_MODEL", "bge-m3-onnx")
        self.chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
        self.bm = np.load(os.path.join(HERE, "bm25.npy"))
        self.vocab = json.load(open(os.path.join(HERE, "bm_vocab.json"), encoding="utf-8"))["vocab"]
        self.vid = {t: i for i, t in enumerate(self.vocab)}
        vec_path = os.environ.get("KB_VEC") or self._default_vec()
        if not os.path.isabs(vec_path):
            # AGENTS.md 里的 zh CLI 写的是裸名 KB_VEC=vec.npy，旧实现按 CWD 解析，
            # 从别的目录跑就会取到主库那份（或 FileNotFound）。相对名一律按库目录解析。
            vec_path = os.path.join(HERE, vec_path)
        self.vecs = np.load(vec_path).astype(np.float32) if os.path.exists(vec_path) else None
        # 三件（chunks / bm25 / vec）必须行对齐：历史上 N 不一致导致 broadcast 崩溃或
        # 静默只排前 N 块。宁可启动即报错，也不能给出看似正常的错分数。
        if self.bm.shape[0] != len(self.chunks):
            raise RuntimeError(f"index mismatch: bm25 rows={self.bm.shape[0]} chunks={len(self.chunks)} ({HERE})")
        if self.vecs is not None and self.vecs.shape[0] != len(self.chunks):
            raise RuntimeError(f"index mismatch: vec rows={self.vecs.shape[0]} chunks={len(self.chunks)} "
                               f"({vec_path} 与 {CHUNKS} 不同代；换 KB_VEC 或重建)")
        if self.vecs is not None:
            self.vecs /= (np.linalg.norm(self.vecs, axis=1, keepdims=True) + 1e-9)
        self._embedder = embedder
        self._alias_map = None

    def _expand(self, query):
        """别名扩写：只作用于 BM25 词面档，不扰动向量查询文本。alias_map.json 存在即生效。"""
        if self._alias_map is None:
            p = os.path.join(HERE, "alias_map.json")
            self._alias_map = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        add = [v for k, v in self._alias_map.items() if k in query]
        return query + (" " + " ".join(add) if add else "")

    def embedder(self):
        if self._embedder is None:
            model = self.model
            if model == "bge-m3-onnx":
                from m3embed import BgeM3ONNX
                self._embedder = BgeM3ONNX()
            else:
                from fastembed import TextEmbedding
                self._embedder = TextEmbedding(model)
        return self._embedder

    @staticmethod
    def _minmax(s, topn=100):
        top = np.argsort(-s)[:topn]
        t = s[top]
        lo, hi = t.min(), t.max()
        out = np.zeros_like(s)
        out[top] = (t - lo) / (hi - lo + 1e-9)
        return out

    def reranker(self):
        if not hasattr(self, "_rerank"):
            from rerank import Reranker
            self._rerank = Reranker()
        return self._rerank

    def query_vector(self, query):
        qi, _ = instr_for(self.model)
        q = np.vstack(list(self.embedder().embed([qi + query])))
        q = q / (np.linalg.norm(q) + 1e-9)
        return q[0]

    def search(self, query, k=8, mode="hybrid", qv=None):
        if mode.endswith("-r"):  # 粗排20 → 精排
            base = {"hybrid-r": "hybrid", "bm25-r": "bm25", "vec-r": "vec"}[mode]
            cand = self.search(query, k=int(os.environ.get("KB_R_COARSE", "20")), mode=base, qv=qv)
            pairs = self.reranker().scores(query, [c["text"] for c, _ in cand])
            order = sorted(range(len(cand)), key=lambda i: -pairs[i])[:k]
            return [(cand[i][0], float(pairs[i])) for i in order]
        bm_scores = np.zeros(len(self.chunks), dtype=np.float32)
        for t in set(tokenize(self._expand(query))):
            if t in self.vid:
                bm_scores += self.bm[:, self.vid[t]]
        if mode in ("vec", "hybrid", "hybrid-v2"):
            if self.vecs is None:
                raise RuntimeError(f"向量档不可用：{HERE} 缺 vec 文件（mode={mode}）")
            if qv is None:
                qv = self.query_vector(query)
            vec_scores = self.vecs @ qv
        if mode == "bm25":
            order = np.argsort(-bm_scores)[:k]
            return [(self.chunks[i], float(bm_scores[i])) for i in order]
        if mode == "vec":
            order = np.argsort(-vec_scores)[:k]
            return [(self.chunks[i], float(vec_scores[i])) for i in order]
        if mode == "hybrid-v2":  # min-max 加权（默认 0.5:0.5；KB_HYB_W 调 BM25 侧权重，供受控扫描）
            bm_mm = self._minmax(bm_scores)
            vc_mm = self._minmax(vec_scores)
            w = float(os.environ.get("KB_HYB_W", "0.5"))
            fused = w * bm_mm + (1 - w) * vc_mm
            order = np.argsort(-fused)[:k]
            return [(self.chunks[i], float(fused[i])) for i in order]
        # hybrid: RRF 融合
        r = {}
        for rank, i in enumerate(np.argsort(-bm_scores)[:40]):
            r[int(i)] = r.get(int(i), 0) + 1 / (60 + rank)
        for rank, i in enumerate(np.argsort(-vec_scores)[:40]):
            r[int(i)] = r.get(int(i), 0) + 1 / (60 + rank)
        top = sorted(r.items(), key=lambda x: -x[1])[:k]
        return [(self.chunks[i], s) for i, s in top]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default=None)
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--mode", default="hybrid-v2", choices=["hybrid", "hybrid-v2", "bm25", "vec", "hybrid-r", "bm25-r", "vec-r"])
    ap.add_argument("--serve", action="store_true", help="常驻模式：每行一个查询，输出 JSON 行")
    a = ap.parse_args()
    jieba.setLogLevel(60)
    kb = KB()
    if a.serve:
        print(json.dumps({"ready": True}), flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                spec = json.loads(line)
            except json.JSONDecodeError:
                spec = {"query": line}
            res = kb.search(spec.get("query", ""), spec.get("k", a.k), spec.get("mode", a.mode))
            print(json.dumps([{"file": c["file"], "line": c["line"], "heading": c["heading"],
                               "score": round(float(s), 4), "text": c["text"]} for c, s in res],
                             ensure_ascii=False), flush=True)
        return
    if not a.query:
        ap.error("需要 query 或 --serve")
    for rank, (c, s) in enumerate(kb.search(a.query, a.k, a.mode), 1):
        snippet = c["text"].replace("\n", " ")[:120]
        print(f"{rank:2d} [{s:.3f}] {c['file']} L{c['line']} #{c['heading'][:40]}\n    {snippet}")

if __name__ == "__main__":
    main()
