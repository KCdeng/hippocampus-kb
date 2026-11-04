"""B 组失败测试：三条直写生产索引的路径必须是"临时件 → 写完 → 原子换档 → 带时间戳留档"。

实测缺口（2026-11-04，崩溃/断电场景盘点）：
  chunker.py 落盘是 `open(args.out, "w")` 直写主库 chunks.jsonl；
  build_vec.py 是 `np.save(a.out, vec)` 直写 vec.npy，provenance 再单独 `open(..., "w")`；
  index.py 直写 bm25.npy / bm_vocab.json / meta.json。
  而 kb_epilogue.py 的复盘区入库链、kb_remember.py 的 mem 重建链，正是拿这三个脚本直写生产件
  （epi/chunks.jsonl、epi/vec.npy）。`open(w)` 与 `np.save` 都不原子：磁盘满、进程被杀、
  Ctrl-C 落在写的中间，生产档就成半截文件——行对齐校验会拒挂它（好），但上一版已经没了（坏）。
  mem 区早就在用 promote_vec 那套（临时 .new.<pid> → 校验 → os.replace → .prev.<ts> 归档），
  本轮把它抽成 kb_atomic 公共件，三条路径共用同一份实现。
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

KB = os.path.dirname(os.path.abspath(__file__))
# 子进程用当前解释器，不写死 venv/Scripts/python.exe：那份路径换台机器（或换个 OS）就
# FileNotFoundError，而这三条用例的判据是"脚本的换档行为"，与谁在跑它无关。
PY = sys.executable
sys.path.insert(0, KB)


def _mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _txt(p):
    with open(p, encoding="utf-8") as r:
        return r.read()


def _w(p, text):
    with open(p, "w", encoding="utf-8") as w:
        w.write(text)


def _jw(p, obj):
    with open(p, "w", encoding="utf-8") as w:
        json.dump(obj, w)


def _jr(p):
    with open(p, encoding="utf-8") as r:
        return json.load(r)


class AtomicStageTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="atomic_")
        self.at = _mod(os.path.join(KB, "kb_atomic.py"), "at_under_test")
        self.dst = os.path.join(self.d, "chunks.jsonl")

    def _residue(self):
        return sorted(f for f in os.listdir(self.d) if ".new." in f)

    def _archives(self):
        return sorted(f for f in os.listdir(self.d)
                      if ".prev." in f and not f.endswith(".provenance.json"))

    def test_stage_promotes_and_archives_previous(self):
        _w(self.dst, "上一版\n")
        with self.at.stage(self.dst) as tmp:
            _w(tmp, "新一版\n")
        self.assertEqual(_txt(self.dst), "新一版\n")
        self.assertEqual(len(self._archives()), 1, f"旧档没带时间戳留档：{sorted(os.listdir(self.d))}")
        self.assertEqual(_txt(os.path.join(self.d, self._archives()[0])), "上一版\n")
        self.assertEqual(self._residue(), [], f"临时件没清走：{self._residue()}")

    def test_failure_inside_stage_leaves_production_untouched(self):
        """崩溃/磁盘满/Ctrl-C 落在写的中间那一瞬：生产档必须还是完整的上一版。"""
        _w(self.dst, "好档\n")
        with self.assertRaises(ValueError):
            with self.at.stage(self.dst) as tmp:
                _w(tmp, "写到一半")
                raise ValueError("模拟中途炸")
        self.assertEqual(_txt(self.dst), "好档\n", "半截覆写进了生产档")
        self.assertEqual(self._residue(), [], f"失败后临时件没清：{self._residue()}")

    def test_sidecar_follows_the_artifact(self):
        """vec 与它的 provenance 必须同一瞬换档：只换 vec，生产档就配着上一版的溯源。"""
        _w(self.dst, "旧\n")
        _jw(self.dst + ".provenance.json", {"model": "old"})
        with self.at.stage(self.dst) as tmp:
            _w(tmp, "新\n")
            _jw(tmp + ".provenance.json", {"model": "bge-m3-onnx"})
        self.assertEqual(_jr(self.dst + ".provenance.json")["model"], "bge-m3-onnx")
        self.assertEqual(len(self._archives()), 1)
        self.assertEqual(_jr(os.path.join(self.d, self._archives()[0]) + ".provenance.json")["model"],
                         "old", "旧溯源没跟着旧档进归档")

    def test_missing_new_sidecar_clears_stale_one(self):
        _w(self.dst, "旧\n")
        _jw(self.dst + ".provenance.json", {"model": "old"})
        with self.at.stage(self.dst) as tmp:
            _w(tmp, "新\n")
        self.assertFalse(os.path.exists(self.dst + ".provenance.json"),
                         "新档没有溯源却留着旧的 = 假溯源")

    def test_np_save_suffix_convention(self):
        """np.save 会给无 .npy 后缀的名字补上 .npy；临时件不同名就找不到它，只能留孤儿。"""
        import numpy as np
        vec = os.path.join(self.d, "vec.npy")
        with self.at.stage(vec) as tmp:
            self.assertTrue(tmp.endswith(".npy"), f"临时名没保住 .npy 后缀：{tmp}")
            np.save(tmp, np.ones((2, 4), dtype=np.float32))
        self.assertEqual(np.load(vec).shape, (2, 4))
        self.assertEqual(self._residue(), [], f"np.save 补后缀导致残留：{self._residue()}")

    def test_promote_prunes_archives_to_keep(self):
        import numpy as np
        vec = os.path.join(self.d, "p.npy")
        for i in range(5):
            tmp = f"{vec}.new.99{i}.npy"
            np.save(tmp, np.ones((i + 1, 4), dtype=np.float32))
            _jw(tmp + ".provenance.json", {"i": i})
            self.at.promote(tmp, vec, keep=3)
        self.assertEqual(len(self._archives()), 3, f"归档没受 keep 约束：{self._archives()}")
        side = sorted(f for f in os.listdir(self.d)
                      if ".prev." in f and f.endswith(".provenance.json"))
        self.assertEqual(len(side), 3, f"归档的溯源没成对剪，磁盘会单向增长：{side}")
        self.assertEqual(self._residue(), [], f"孤儿临时件没收干净：{self._residue()}")

    def test_same_second_promotes_do_not_eat_each_other(self):
        import numpy as np
        vec = os.path.join(self.d, "q.npy")
        for i in range(4):
            tmp = f"{vec}.new.7{i}.npy"
            np.save(tmp, np.ones((i + 1, 4), dtype=np.float32))
            self.at.promote(tmp, vec, keep=3)
        rows = sorted(int(np.load(os.path.join(self.d, f)).shape[0]) for f in self._archives())
        self.assertEqual(rows, [1, 2, 3], f"同一秒内连续换档把中间归档吃掉了：{rows}")


class ChunkerEmitTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="chunkemit_")
        self.ch = _mod(os.path.join(KB, "chunker.py"), "chunker_under_test")
        self.out = os.path.join(self.d, "chunks.jsonl")

    def _residue(self):
        return sorted(f for f in os.listdir(self.d) if ".new." in f)

    def test_emit_writes_rows(self):
        self.ch.emit([{"id": "a::L1::0", "file": "a.md", "line": 1, "heading": "h",
                       "text": "正文", "chars": 2, "kind": "text"}], self.out)
        self.assertEqual(len(_txt(self.out).splitlines()), 1)
        self.assertEqual(self._residue(), [])

    def test_emit_failure_keeps_previous_production_file(self):
        """序列化中途抛（真实诱因：某块带了不可序列化的值）→ 生产档必须还是上一版全文。"""
        good = '{"file":"old.md","text":"上一版整份"}\n' * 3
        _w(self.out, good)
        rows = [{"file": "a.md"}, {"file": "b.md", "text": {1, 2}}]   # set 不可 JSON 序列化
        with self.assertRaises(TypeError):
            self.ch.emit(rows, self.out)
        self.assertEqual(_txt(self.out), good, "半截 chunks.jsonl 覆写了生产档")
        self.assertEqual(self._residue(), [], f"失败后留下临时件：{self._residue()}")


class ChunkerCliTest(unittest.TestCase):
    """真实调用姿势：kb_epilogue / kb_remember 是拿 chunker.py --out 直写区索引的。"""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="chunkcli_")
        self.root = os.path.join(self.d, "corpus")
        os.makedirs(self.root)
        _w(os.path.join(self.root, "note.md"),
           "# 标题\n\n" + "这一段正文要够长才能被切成块，重复它几次。" * 6 + "\n")
        self.out = os.path.join(self.d, "chunks.jsonl")

    def _run(self):
        return subprocess.run([PY, "-X", "utf8", os.path.join(KB, "chunker.py"),
                               "--root", self.root, "--out", self.out, "--force"],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=180)

    def test_two_runs_reproducible_and_no_residue(self):
        r1 = self._run()
        self.assertEqual(r1.returncode, 0, r1.stderr[-400:])
        first = _txt(self.out)
        r2 = self._run()
        self.assertEqual(r2.returncode, 0, r2.stderr[-400:])
        self.assertEqual(_txt(self.out), first, "同源同参重切应当逐字节可复现")
        self.assertEqual(sorted(f for f in os.listdir(self.d) if ".new." in f), [],
                         "临时件没原子换档干净")
        self.assertEqual(len([f for f in os.listdir(self.d) if "chunks.jsonl.prev." in f]), 1,
                         f"上一版没带时间戳留档：{sorted(os.listdir(self.d))}")


class BuildVecSaveTest(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.d = tempfile.mkdtemp(prefix="bvec_")
        self.bv = _mod(os.path.join(KB, "build_vec.py"), "build_vec_under_test")
        self.vec = os.path.join(self.d, "vec.npy")
        np.save(self.vec, np.ones((2, 8), dtype=np.float32))
        _jw(self.vec + ".provenance.json", {"model": "old"})

    def _residue(self):
        return sorted(f for f in os.listdir(self.d) if ".new." in f)

    def test_save_vec_lands_pair(self):
        import numpy as np
        self.bv.save_vec(np.zeros((3, 8), dtype=np.float32), self.vec,
                         {"model": "bge-m3-onnx", "n_chunks": 3})
        self.assertEqual(np.load(self.vec).shape, (3, 8))
        self.assertEqual(_jr(self.vec + ".provenance.json")["n_chunks"], 3)
        self.assertEqual(self._residue(), [])

    def test_save_vec_failure_keeps_production_vec(self):
        import numpy as np
        with self.assertRaises(TypeError):
            self.bv.save_vec(np.zeros((3, 8), dtype=np.float32), self.vec, {"sha": {1, 2}})
        self.assertEqual(np.load(self.vec).shape, (2, 8), "半截向量档覆写了生产 vec")
        self.assertEqual(_jr(self.vec + ".provenance.json")["model"], "old")
        self.assertEqual(self._residue(), [], f"失败后留下临时件：{self._residue()}")


class IndexAtomicTest(unittest.TestCase):
    """index.py 一次写 bm25.npy / bm_vocab.json / meta.json 三件，每件都得原子落。"""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="idxatomic_")
        rows = [{"id": f"a::L{i}::0", "file": "a.md", "line": i, "heading": "标题",
                 "text": f"第{i}段正文，够长够长够长。", "chars": 12, "kind": "text"}
                for i in range(1, 4)]
        _w(os.path.join(self.d, "chunks.jsonl"),
           "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    def _run(self):
        env = dict(os.environ, KB_DIR=self.d, KB_MODEL="bge-m3-onnx")
        return subprocess.run([PY, "-X", "utf8", os.path.join(KB, "index.py")],
                              env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=300)

    def test_build_writes_three_artifacts_atomically(self):
        r = self._run()
        self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-500:])
        for name in ("bm25.npy", "bm_vocab.json", "meta.json"):
            self.assertTrue(os.path.exists(os.path.join(self.d, name)), f"{name} 没落盘")
        self.assertEqual(sorted(f for f in os.listdir(self.d) if ".new." in f), [],
                         "索引构建留下半截临时件")
        self.assertEqual(_jr(os.path.join(self.d, "meta.json"))["n_chunks"], 3)

    def test_second_build_keeps_previous_artifacts(self):
        self.assertEqual(self._run().returncode, 0)
        self.assertEqual(self._run().returncode, 0)
        prev = [f for f in os.listdir(self.d) if ".prev." in f]
        self.assertTrue(prev, f"重建没留下可回滚的上一版：{sorted(os.listdir(self.d))}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
