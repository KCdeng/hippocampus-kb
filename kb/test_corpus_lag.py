"""C 组失败测试：崩溃/中断打断入库后，"语料比索引新"要**报警**——但绝不自动重建。

实测缺口：`kb_remember add` 先写 mem/corpus/mem-0029.md、再重建 mem 三件。进程在两步之间死掉
（Ctrl-C、断电、OOM），那条记忆就静静躺在语料里查不到；而 watcher 只盯**索引档** mtime，
索引没动 → 它永远不会发现。/stats 的 stale 也不报：stale 只比"索引里实际出现的文件"，
没入库的文件压根不在集合里（这是当初为了不让我新写的规程点亮主库灯而定的口径）。

设计约束（用户定的）：尾部区的入库闸门是人走的，自动重建等于把写权限交给 watcher。
所以这里只做检测与报警（STATE["lag"] + /health + /stats + stderr），修不修由人决定。
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest

KB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KB)


def _mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _w(p, text):
    with open(p, "w", encoding="utf-8") as w:
        w.write(text)


class CorpusLagTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="lag_")
        self.serve = _mod(os.path.join(KB, "kb_serve.py"), "serve_lag")
        self.serve.HERE = self.d
        self.serve.WORKSPACE = os.path.join(self.d, "workspace")
        for z in ("mem", "epi"):
            os.makedirs(os.path.join(self.d, z, "corpus"), exist_ok=True)
        self.zone = lambda z, name="chunks.jsonl": os.path.join(self.d, z, name)
        self.corpus = lambda z: os.path.join(self.d, z, "corpus")

    def _chunk(self, zone, fname):
        with open(self.zone(zone), "a", encoding="utf-8") as w:
            w.write(json.dumps({"file": fname, "line": 1, "heading": "h", "text": "正文"},
                               ensure_ascii=False) + "\n")

    def test_mem_corpus_file_missing_from_index_is_reported(self):
        _w(os.path.join(self.corpus("mem"), "mem-0001.md"), "---\n---\n旧条目\n")
        _w(os.path.join(self.corpus("mem"), "mem-0002.md"), "---\n---\n崩溃前刚写的那条\n")
        self._chunk("mem", "mem-0001.md")          # 索引只吃到第一条
        got = self.serve.corpus_lag_warnings()
        self.assertEqual(len(got), 1, f"add 写完 md 就崩，索引落后却没人报：{got}")
        self.assertIn("mem", got[0])
        self.assertIn("mem-0002.md", got[0], "报警没点出是哪个文件没入库，等于没法修")

    def test_fully_indexed_zone_is_silent(self):
        _w(os.path.join(self.corpus("mem"), "mem-0001.md"), "---\n---\n一条\n")
        _w(os.path.join(self.corpus("epi"), "epi-0001.md"), "---\n---\n一段\n")
        self._chunk("mem", "mem-0001.md")
        self._chunk("epi", "epi-0001.md")
        self.assertEqual(self.serve.corpus_lag_warnings(), [])

    def test_missing_index_with_nonempty_corpus_is_reported(self):
        _w(os.path.join(self.corpus("epi"), "epi-0007.md"), "---\n---\n入库前就崩了\n")
        got = self.serve.corpus_lag_warnings()
        self.assertEqual(len(got), 1, f"该区一条都查得到却没有任何索引，必须报：{got}")
        self.assertIn("epi", got[0])

    def test_empty_zone_stays_silent(self):
        """全新空区不该报警——没有语料就没有落后。"""
        _w(self.zone("mem"), "")
        _w(self.zone("epi"), "")
        self.assertEqual(self.serve.corpus_lag_warnings(), [])

    def test_main_zone_lag_is_not_reported(self):
        """主库故意不查：工作区里新写而没入库的笔记/规程是常态，报出来只会变成永久噪声。
        （这条口径和 /stats 的 stale 判定一致。）"""
        root = self.serve.WORKSPACE
        os.makedirs(root, exist_ok=True)
        _w(os.path.join(root, "fresh_note.md"), "# 新写的笔记\n\n正文正文正文。\n")
        _w(os.path.join(self.d, "chunks.jsonl"),          # 主库索引档在 kb/ 根上
           json.dumps({"file": "already.md", "line": 1, "heading": "h", "text": "正文"},
                      ensure_ascii=False) + "\n")
        self.assertEqual(self.serve.corpus_lag_warnings(), [])


class LagAlertingTest(unittest.TestCase):
    """报警落地：/health 与 /stats 都要带 lag，光落盘不露出等于没报警。"""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="lagexpose_")
        self.serve = _mod(os.path.join(KB, "kb_serve.py"), "serve_lag_expose")
        self.serve.HERE = self.d
        os.makedirs(os.path.join(self.d, "mem", "corpus"), exist_ok=True)
        _w(os.path.join(self.d, "mem", "corpus", "mem-0009.md"), "---\n---\n未入库\n")

    def test_lag_refresh_updates_state(self):
        self.serve.STATE["lag"] = []
        self.serve.refresh_lag_alerts()
        self.assertEqual(len(self.serve.STATE["lag"]), 1, "刷新后 STATE 里没有 lag")
        self.assertIn("mem-0009.md", self.serve.STATE["lag"][0])
        self.assertIn(self.serve.STATE["lag"][0], self.serve.STATE["warn"],
                      "lag 没并进 warn——现有健康探测只看 warn，藏在旁边就看不见")
        # 再刷一次不重复计数（否则 warn 列表会被同一条报警撑爆）
        self.serve.refresh_lag_alerts()
        self.assertEqual(len(self.serve.STATE["lag"]), 1)
        self.assertEqual(self.serve.STATE["warn"].count(self.serve.STATE["lag"][0]), 1)

    def test_recovered_zone_clears_alert(self):
        self.serve.refresh_lag_alerts()
        first = self.serve.STATE["lag"][0]
        with open(os.path.join(self.d, "mem", "chunks.jsonl"), "w", encoding="utf-8") as w:
            w.write(json.dumps({"file": "mem-0009.md", "line": 1, "heading": "h", "text": "已入库"})
                    + "\n")
        self.serve.refresh_lag_alerts()
        self.assertEqual(self.serve.STATE["lag"], [], "重建后报警没撤，灯会一直黄着")
        self.assertNotIn(first, self.serve.STATE["warn"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
