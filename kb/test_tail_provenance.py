"""尾部区（mem/epi）向量溯源护栏的失败测试。

背景：main/zh/comp 都有 vec.npy.provenance.json 且启动时校验模型名；mem/epi 没有。
实测缺口有两个：
  1. kb_remember.rebuild_mem() 把 vec.npy.new.<pid>.npy 原子改名成 vec.npy，
     但 build_vec.py 写在临时名旁边的 provenance sidecar 没跟着改名 ——
     生产 vec.npy 永远缺溯源，磁盘上反而堆孤儿 sidecar（实测 4 份）。
  2. 8310 与 Hermes MCP 的启动校验只覆盖 main/zh/comp，尾部区换模型不会报警。
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest

KB = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(KB)
sys.path.insert(0, KB)

HERMES_MCP = os.environ.get("HERMES_MCP", "")


def _hermes_mcp():
    """McpTailProvenanceTest 验的是**对端**（Hermes 共用同一引擎）的拒绝挂载行为。
    对端不在这台机器上就跳过，而不是让整组守门员在别人的机器上报错。"""
    if not os.path.exists(HERMES_MCP):
        raise unittest.SkipTest(f"未找到对端 mcp_server.py（可用 HERMES_MCP 指定）：{HERMES_MCP}")
    return HERMES_MCP


def _mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fake_fastmcp():
    """mcp_server 顶部 import fastembed/fastmcp；测试只用到纯 Python 校验路径。"""
    if "mcp" in sys.modules:
        return
    pkg = type(sys)("mcp"); server = type(sys)("mcp.server"); fast = type(sys)("mcp.server.fastmcp")

    class FastMCP:
        def __init__(self, *a, **k): self._r = []
        def resource(self, *a, **k): return lambda f: f
        def prompt(self, *a, **k): return lambda f: f
        def tool(self, *a, **k): return lambda f: f
        def run(self, *a, **k): pass

    fast.FastMCP = FastMCP; pkg.server = server; server.fastmcp = fast
    sys.modules.update({"mcp": pkg, "mcp.server": server, "mcp.server.fastmcp": fast})


class PromoteVecTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="memvec_")
        self.kr = _mod(os.path.join(KB, "kb_remember.py"), "kr_under_test")
        self.vec = os.path.join(self.d, "vec.npy")

    def _tmp(self, pid=999, rows=3):
        import numpy as np
        tmp = self.vec + f".new.{pid}.npy"
        np.save(tmp, np.zeros((rows, 4), dtype=np.float32))
        json.dump({"model": "bge-m3-onnx", "n_chunks": rows,
                   "chunks": os.path.join(self.d, "chunks.jsonl")},
                  open(tmp + ".provenance.json", "w", encoding="utf-8"))
        return tmp

    def test_promote_moves_sidecar_with_vec(self):
        tmp = self._tmp()
        self.kr.promote_vec(tmp, self.vec)
        self.assertTrue(os.path.exists(self.vec))
        p = self.vec + ".provenance.json"
        self.assertTrue(os.path.exists(p), "vec 改名了但 sidecar 还留在临时名上 → 生产档无溯源")
        self.assertEqual(json.load(open(p, encoding="utf-8"))["n_chunks"], 3)
        self.assertFalse(os.path.exists(tmp), "临时 vec 未清走")
        self.assertFalse(os.path.exists(tmp + ".provenance.json"), "孤儿临时 sidecar 未清走")

    def test_promote_prunes_older_orphans(self):
        old = self._tmp(pid=111)
        os.rename(old + ".provenance.json", self.vec + ".new.111.npy.provenance.json")
        if os.path.exists(old):
            os.remove(old)
        tmp = self._tmp(pid=222)
        self.kr.promote_vec(tmp, self.vec)
        left = [f for f in os.listdir(self.d) if ".new." in f]
        self.assertEqual(left, [], f"历史孤儿没被收干净：{left}")
        self.assertTrue(os.path.exists(self.vec + ".provenance.json"))

    def test_promote_without_sidecar_leaves_no_stale_claim(self):
        """临时 sidecar 缺失时，旧的生产 sidecar 描述的是上一版向量，必须一起清掉，
        否则溯源会把旧档的模型/哈希当成当前 vec.npy 的。"""
        json.dump({"model": "bge-small-zh-v1.5", "n_chunks": 99},
                  open(self.vec + ".provenance.json", "w", encoding="utf-8"))
        tmp = self._tmp()
        os.remove(tmp + ".provenance.json")
        self.kr.promote_vec(tmp, self.vec)
        self.assertTrue(os.path.exists(self.vec))
        self.assertFalse(os.path.exists(self.vec + ".provenance.json"),
                         "上一版溯源被留给了新 vec —— 假的溯源比没有溯源更糟")


class ServeTailProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="tailprov_")
        self.serve = _mod(os.path.join(KB, "kb_serve.py"), "serve_under_test")
        self.orig = self.serve.HERE
        self.serve.HERE = self.d
        for z in ("mem", "epi"):
            os.makedirs(os.path.join(self.d, z))
            open(os.path.join(self.d, z, "chunks.jsonl"), "w").close()
            os.makedirs(os.path.join(self.d, z, "corpus"))

    def tearDown(self):
        self.serve.HERE = self.orig

    def _prov(self, zone, model):
        json.dump({"model": model}, open(os.path.join(self.d, zone, "vec.npy.provenance.json"),
                                         "w", encoding="utf-8"))

    def test_mismatch_is_reported(self):
        self._prov("mem", "bge-small-zh-v1.5")
        w = self.serve.tail_prov_warnings()
        self.assertEqual(len(w), 1, f"尾部区模型不对却没报警：{w}")
        self.assertIn("mem", w[0])

    def test_matching_model_and_missing_sidecar_are_silent(self):
        self._prov("mem", self.serve.MODEL_MAIN)   # epi 没 sidecar（历史档）
        self.assertEqual(self.serve.tail_prov_warnings(), [])


class McpTailProvenanceTest(unittest.TestCase):
    def setUp(self):
        _fake_fastmcp()
        self.d = tempfile.mkdtemp(prefix="mcptail_")
        os.makedirs(os.path.join(self.d, "kb", "mem"))
        open(os.path.join(self.d, "kb", "mem", "chunks.jsonl"), "w").close()
        self.srv = _mod(_hermes_mcp(), "mcp_under_test")
        self.srv.KB_DIR = self.d
        self.srv._state["mem"] = None

    def test_mem_refused_when_model_mismatch(self):
        json.dump({"model": "intfloat/multilingual-e5-large"},
                  open(os.path.join(self.d, "kb", "mem", "vec.npy.provenance.json"),
                       "w", encoding="utf-8"))
        with self.assertRaises(RuntimeError) as cm:
            self.srv.get_mem()
        self.assertIn("mem", str(cm.exception))

    def test_mem_mounts_when_sidecar_absent(self):
        """历史 mem 档没有 sidecar，不能因此拒绝挂载（只由 8310/健康探测报缺）。"""
        self.srv.load_kb = lambda d, m, v: (type("KB", (), {})(), None)
        self.srv._get_m3 = lambda: None
        self.assertIsNotNone(self.srv.get_mem())


class RebuildZeroGuardTest(unittest.TestCase):
    """隔离演练发现：mem 语料若被清空/切块异常，chunker 会写出 0 行 chunks.jsonl，
    build_vec 跟着产出 (0, 1024) 的 vec，行数校验 0==0 通过 → 生产记忆区被静默清空。
    重建必须先把块数守住，再动生产件。"""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="memzero_")
        self.kr = _mod(os.path.join(KB, "kb_remember.py"), "kr_zero")
        self.MEM = os.path.join(self.d, "mem")
        os.makedirs(os.path.join(self.MEM, "corpus"))
        self.chunks = os.path.join(self.MEM, "chunks.jsonl")
        self.vec = os.path.join(self.MEM, "vec.npy")
        open(self.chunks, "w", encoding="utf-8").write(
            '{"file":"mem-0001.md","heading":"h","text":"既有的生产记忆块内容够长够长够长。\n"}\n'
            '{"file":"mem-0002.md","heading":"h","text":"另一条既有生产记忆，够长够长够长。\n"}\n')
        import numpy as np
        np.save(self.vec, np.ones((2, 8), dtype=np.float32))
        json.dump({"model": "bge-m3-onnx", "n_chunks": 2},
                  open(self.vec + ".provenance.json", "w", encoding="utf-8"))
        self.kr.MEM = self.MEM
        self.kr.CORPUS = os.path.join(self.MEM, "corpus")

    def tearDown(self):
        import subprocess
        subprocess.run = self._real_run

    def _stub(self, rows_written):
        import subprocess
        self._real_run = subprocess.run

        def fake_run(cmd, *a, **k):
            if cmd[1].endswith("chunker.py"):
                out = cmd[cmd.index("--out") + 1]
                open(out, "w", encoding="utf-8").write("".join(
                    '{"file":"mem-9001.md","heading":"h","text":"x"}\n' for _ in range(rows_written)))
            elif cmd[1].endswith("build_vec.py"):
                import numpy as np
                out = cmd[cmd.index("--out") + 1]
                np.save(out, np.ones((rows_written, 8), dtype=np.float32))
                json.dump({"model": "bge-m3-onnx", "n_chunks": rows_written},
                          open(out + ".provenance.json", "w", encoding="utf-8"))
            return subprocess.CompletedProcess(cmd, 0)

        self.kr.subprocess.run = fake_run

    def test_zero_row_rechunk_keeps_production_intact(self):
        before = open(self.chunks, encoding="utf-8").read()
        import numpy as np
        self._stub(0)
        with self.assertRaises(SystemExit) as cm:
            self.kr.rebuild_mem()
        self.assertIn("0", str(cm.exception))
        self.assertEqual(open(self.chunks, encoding="utf-8").read(), before, "空切块覆写了生产 chunks.jsonl")
        self.assertEqual(np.load(self.vec).shape, (2, 8), "空重建覆写了生产 vec.npy")
        self.assertEqual(json.load(open(self.vec + ".provenance.json", encoding="utf-8"))["n_chunks"], 2)

    def test_normal_rebuild_still_promotes(self):
        self._stub(3)
        import numpy as np
        self.kr.rebuild_mem()
        self.assertEqual(np.load(self.vec).shape, (3, 8))
        self.assertEqual(json.load(open(self.vec + ".provenance.json", encoding="utf-8"))["n_chunks"], 3)
        self.assertEqual(len([l for l in open(self.chunks, encoding="utf-8") if l.strip()]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
