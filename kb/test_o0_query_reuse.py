import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

KB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(KB_DIR))

from fuse import Fused
from kb_search import KB

HERMES_MCP = os.environ.get("HERMES_MCP", "")


def _hermes_mcp():
    """这两条用例验的是**对端**（Hermes 共用同一引擎）的调用次数。对端不在这台机器上就跳过，
    而不是让整套守门员在没有对端的机器上报错。"""
    if not os.path.exists(HERMES_MCP):
        raise unittest.SkipTest(f"未找到对端 mcp_server.py（可用 HERMES_MCP 指定）：{HERMES_MCP}")
    return HERMES_MCP


class CountingEmbedder:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)
        self.calls = []

    def embed(self, texts):
        texts = list(texts)
        self.calls.extend(texts)
        return [self.vector.copy() for _ in texts]


def make_kb(zone, embedder, model="bge-m3-onnx"):
    kb = KB.__new__(KB)
    kb.model = model
    kb.chunks = [
        {"id": f"{zone}-0", "file": f"{zone}.md", "line": 1,
         "heading": zone, "text": "alpha"},
        {"id": f"{zone}-1", "file": f"{zone}.md", "line": 2,
         "heading": zone, "text": "beta"},
    ]
    kb.bm = np.zeros((2, 1), dtype=np.float32)
    kb.vocab = ["alpha"]
    kb.vid = {"alpha": 0}
    kb.vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    kb._embedder = embedder
    kb._alias_map = {}
    return kb


def load_mcp_server():
    fastmcp = types.ModuleType("mcp.server.fastmcp")

    class FastMCP:
        def __init__(self, _name):
            self.settings = types.SimpleNamespace()

        def tool(self):
            return lambda fn: fn

    fastmcp.FastMCP = FastMCP
    modules = {
        "mcp": types.ModuleType("mcp"),
        "mcp.server": types.ModuleType("mcp.server"),
        "mcp.server.fastmcp": fastmcp,
    }
    path = Path(_hermes_mcp())
    spec = importlib.util.spec_from_file_location("o0_mcp_server_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class QueryVectorTest(unittest.TestCase):
    def test_query_vector_applies_model_instruction_and_normalizes(self):
        embedder = CountingEmbedder([3.0, 4.0])
        kb = make_kb("zh", embedder, "intfloat/multilingual-e5-large")

        self.assertTrue(hasattr(kb, "query_vector"), "KB.query_vector 尚不存在")
        qv = kb.query_vector("急停")

        self.assertEqual(embedder.calls, ["query: 急停"])
        np.testing.assert_allclose(qv, [0.6, 0.8], atol=1e-6)
        self.assertEqual(qv.shape, (2,))

    def test_search_uses_supplied_vector_without_embedding(self):
        embedder = CountingEmbedder([1.0, 0.0])
        kb = make_kb("main", embedder)
        qv = np.array([0.0, 1.0], dtype=np.float32)

        try:
            result = kb.search("beta", k=2, mode="vec", qv=qv)
        except TypeError as exc:
            self.fail(f"KB.search 尚未支持 qv 透传: {exc}")

        self.assertEqual(embedder.calls, [])
        self.assertEqual([c["id"] for c, _ in result], ["main-1", "main-0"])
        self.assertAlmostEqual(result[0][1], 1.0, places=6)
        self.assertAlmostEqual(result[1][1], 0.0, places=6)


class FusedReuseTest(unittest.TestCase):
    def test_main_mem_epi_embed_once_and_preserve_results(self):
        embedder = CountingEmbedder([1.0, 0.0])
        doc = make_kb("main", embedder)
        mem = make_kb("mem", embedder)
        epi = make_kb("epi", embedder)
        fused = Fused.__new__(Fused)
        fused.doc = doc
        fused.tails = [
            {"name": "mem", "kb": mem, "flags": {}},
            {"name": "epi", "kb": epi, "flags": {}},
        ]
        query = "急停"

        docs = doc.search(query, k=2, mode="hybrid-v2")
        mem_hits = mem.search(query, k=2, mode="hybrid-v2")
        epi_hits = epi.search(query, k=2, mode="hybrid-v2")
        expected = ([("doc", c, s) for c, s in docs]
                    + [("mem", c, s) for c, s in mem_hits]
                    + [("epi", c, s) for c, s in epi_hits])
        embedder.calls = []

        actual = fused.search(query, k=2, layout="tiered", mode="hybrid-v2")

        self.assertEqual([(z, c["id"]) for z, c, _ in actual],
                         [(z, c["id"]) for z, c, _ in expected])
        for (_, _, actual_score), (_, _, expected_score) in zip(actual, expected):
            self.assertAlmostEqual(actual_score, expected_score, delta=1e-6)
        self.assertEqual(len(embedder.calls), 1,
                         "main/mem/epi 应在请求内复用同一次 m3 查询嵌入")


class MCPReuseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = load_mcp_server()

    def setUp(self):
        self.m3 = CountingEmbedder([1.0, 0.0])
        self.e5 = CountingEmbedder([0.0, 1.0])
        self.zones = {
            "main": make_kb("main", self.m3),
            "mem": make_kb("mem", self.m3),
            "epi": make_kb("epi", self.m3),
            "comp": make_kb("comp", self.m3),
            "zh": make_kb("zh", self.e5, "intfloat/multilingual-e5-large"),
        }
        self.old_flags = self.server._state["flags"]
        self.server._state["flags"] = {}
        self.addCleanup(self._restore_flags)

    def _restore_flags(self):
        self.server._state["flags"] = self.old_flags

    def _expected_rows(self, query, k, corpus):
        heads = {"main": ["main"], "all": ["main", "zh", "comp"]}[corpus]
        zone_lists = [(z, self.zones[z].search(query, k=k, mode=self.server._MODES[z]))
                      for z in heads]
        if len(zone_lists) > 1:
            merged = self.server._rrf_merge(zone_lists, k)
        else:
            origin, hits = zone_lists[0]
            merged = [(origin, c, s) for c, s in hits]
        rows = [self.server._row(i + 1, origin, c, s)
                for i, (origin, c, s) in enumerate(merged)]
        for zone in ("mem", "epi"):
            for c, s in self.zones[zone].search(query, k=min(2, k),
                                                mode=self.server._MODES[zone]):
                rows.append(self.server._row(len(rows) + 1, zone, c, s))
        return rows

    def _run_search(self, corpus):
        getters = {f"get_{zone}": (lambda zone=zone: self.zones[zone])
                   for zone in self.zones}
        with patch.multiple(self.server, **getters):
            return json.loads(self.server.search_kb("急停", k=2, corpus=corpus))

    def test_main_reuses_one_m3_vector_and_preserves_response(self):
        expected = self._expected_rows("急停", 2, "main")
        self.m3.calls = []

        payload = self._run_search("main")

        self.assertEqual(payload["results"], expected)
        self.assertEqual(len(self.m3.calls), 1,
                         "MCP main 应将 main/mem/epi 的 m3 嵌入从 3 次降为 1 次")

    def test_all_reuses_one_m3_vector_and_keeps_zh_e5_independent(self):
        expected = self._expected_rows("急停", 2, "all")
        self.m3.calls = []
        self.e5.calls = []

        payload = self._run_search("all")

        self.assertEqual(payload["results"], expected)
        self.assertEqual(len(self.m3.calls), 1,
                         "MCP all 应将 main/comp/mem/epi 的 m3 嵌入从 4 次降为 1 次")
        self.assertEqual(len(self.e5.calls), 1,
                         "zh 必须继续独立执行 e5 查询嵌入")


if __name__ == "__main__":
    unittest.main()
