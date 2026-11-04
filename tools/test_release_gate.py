r"""test_release_gate.py — 发布件净化闸门（先红后绿）。

要拦的事故只有一类：把「只有本机该看见的东西」推到公开仓。所以断言分四层：
  1 白名单——发布目录里的文件必须是我们主动选的，语料/索引/日志/记忆文档一件都不许在；
  2 净化——代码里的本地绝对路径与身份字面量必须被替换掉，闸门要能抓到人为种下的泄漏；
  3 可跑——随仓发布的示例金标准必须指向示例语料里真实存在、且原文锚点确实在文中的题，
    否则外人 clone 完跑不了评测，仓库里的数字就成了无法自证的声明；
  4 完整——README / requirements / LICENSE / .gitignore 这些「别人能不能上手」的门面件要在。
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import check_release as CR      # noqa: E402
import pack_release as PR       # noqa: E402


def _pack_once(cache={}):
    """整树打包很贵（要读几十个文件），全组共用一份 dist。"""
    if "dist" not in cache:
        d = tempfile.mkdtemp(prefix="dist_")
        PR.pack(d)
        cache["dist"] = d
    return cache["dist"]


def _read_file(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def _read(dist, *parts):
    return _read_file(os.path.join(dist, *parts))


def _paths(dist):
    out = []
    for root, dirs, files in os.walk(dist):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        for f in files:
            out.append(os.path.relpath(os.path.join(root, f), dist).replace("\\", "/"))
    return sorted(out)


class WhitelistTest(unittest.TestCase):
    def setUp(self):
        self.dist = _pack_once()
        self.paths = _paths(self.dist)

    def test_engine_and_tests_are_present(self):
        for p in ("kb/chunker.py", "kb/index.py", "kb/kb_serve.py", "kb/kb_atomic.py",
                  "kb/_start.ps1", "kb/test_atomic_write.py", "tools/pack_release.py",
                  "tools/check_release.py", "README.md"):
            self.assertIn(p, self.paths, f"发布件缺 {p}")

    def test_corpus_indexes_logs_and_memory_are_never_shipped(self):
        """命中即事故：语料正文、派生索引、查询日志、记忆文档、模型权重、依赖与实验目录。"""
        hits = [p for p in self.paths if CR.is_forbidden_path(p)]
        self.assertEqual(hits, [], f"发布件里出现了禁止随仓发布的路径：{hits}")

    def test_pack_reports_no_missing_whitelist_source(self):
        """白名单点名的源文件若在本地树里不存在，打包必须报出来而不是静默少发一件。"""
        self.assertEqual(PR.pack(tempfile.mkdtemp(prefix="d2_"))["missing"], [])


class SanitizeTest(unittest.TestCase):
    def setUp(self):
        self.dist = _pack_once()

    def test_no_builtin_leak_patterns_anywhere_in_dist(self):
        findings = CR.check(self.dist)
        self.assertEqual([f["detail"] for f in findings], [])

    def test_gate_catches_a_planted_identity_path(self):
        d = tempfile.mkdtemp(prefix="plant_")
        os.makedirs(os.path.join(d, "kb"))
        with open(os.path.join(d, "kb", "tool.py"), "w", encoding="utf-8") as w:
            w.write('SRC = r"Z:\\private\\someone\\desk"\n')
        kinds = [f["kind"] for f in CR.check(d, extra_banned=[r"someone"])]
        self.assertIn("leak", kinds)

    def test_gate_catches_a_planted_forbidden_path(self):
        d = tempfile.mkdtemp(prefix="plant2_")
        os.makedirs(os.path.join(d, "kb"))
        open(os.path.join(d, "kb", "chunks.jsonl"), "w").close()
        open(os.path.join(d, "kb", "bm25.npy"), "w").close()
        kinds = [f["kind"] for f in CR.check(d)]
        self.assertIn("forbidden_path", kinds)

    def test_gate_rejects_oversize_and_binary(self):
        d = tempfile.mkdtemp(prefix="plant3_")
        os.makedirs(os.path.join(d, "kb"))
        with open(os.path.join(d, "kb", "big.md"), "w", encoding="utf-8") as w:
            w.write("x" * (3 * 1024 * 1024))
        with open(os.path.join(d, "kb", "bin.py"), "wb") as w:
            w.write(b"\xc3\x28\x00\x00")        # 非法 UTF-8 字节序列（\x00 本身是合法 UTF-8）
        kinds = {f["kind"] for f in CR.check(d)}
        self.assertIn("oversize", kinds)
        self.assertIn("binary", kinds)

    def test_local_path_literals_are_replaced_not_merged_away(self):
        """替换表漏一行不会报错，只会把本机路径发出去——所以逐处断言替换后的形态。"""
        chunker = _read(self.dist, "kb", "chunker.py")
        self.assertIn('os.environ.get("KB_ROOT"', chunker)
        m3 = _read(self.dist, "kb", "m3embed.py")
        self.assertIn('os.environ.get("M3_ONNX_DIR"', m3)
        serve = _read(self.dist, "kb", "kb_serve.py")
        self.assertIn("python -X utf8 kb_serve.py", serve)


class RunnableExampleTest(unittest.TestCase):
    def setUp(self):
        self.dist = _pack_once()

    def test_example_gold_set_points_at_the_example_corpus(self):
        ev = json.loads(_read(self.dist, "kb", "eval_set.json"))
        self.assertGreaterEqual(len(ev), 12, "示例金标准太少，闸门没有区分度")
        root = os.path.join(self.dist, "corpus")
        for e in ev:
            p = os.path.join(root, e["gold_file"].replace("/", os.sep))
            self.assertTrue(os.path.exists(p), f'{e["qid"]} 的 gold_file 不在示例语料里：{e["gold_file"]}')
            body = "".join(_read_file(p).split())
            self.assertIn("".join(e["gold_anchor"].split()), body,
                          f'{e["qid"]} 的 gold_anchor 在原文里找不到，永远判负')

    def test_example_baseline_has_machine_gate_fields(self):
        b = json.loads(_read(self.dist, "kb", "baseline.json"))
        self.assertIn("gate", b)
        self.assertGreater(b["baseline"]["chunks"], 0)


class FrontageTest(unittest.TestCase):
    def setUp(self):
        self.dist = _pack_once()
        self.paths = _paths(self.dist)

    def test_quickstart_frontage_files_exist(self):
        for p in ("README.md", "requirements.txt", "LICENSE", ".gitignore"):
            self.assertIn(p, self.paths)

    def test_requirements_pin_exactly_the_imported_third_party(self):
        # 这四个是发布件里 grep 出来的全部第三方 import；requests/uvicorn 在 venv 里但没人引用，
        # 钉进去等于让外人装两个用不到的包。
        req = _read(self.dist, "requirements.txt")
        for pkg in ("jieba", "numpy", "onnxruntime", "tokenizers"):
            self.assertIn(pkg, req, f"requirements.txt 未钉住 {pkg}")
        for ghost in ("uvicorn", "requests"):
            self.assertNotIn(ghost, req, f"requirements.txt 钉了没被引用的 {ghost}")

    def test_readme_declares_what_is_not_published(self):
        txt = _read(self.dist, "README.md")
        for token in ("corpus", "记忆", "baseline"):
            self.assertIn(token, txt, f"README 没交代发布边界（缺关键词 {token}）")


class ReadmeLanguagePairTest(unittest.TestCase):
    """About 栏挂 "Language: 简体中文 | English" 就必须真有第二个文件、且互链闭合。

    要拦的事故：描述里写了语言切换，链接却指向不存在的文件——外人第一屏点进 404，
    比根本没有切换栏更糟。另一半事故是英文版写成一份"精简宣传页"，与中文版口径漂移。
    """

    def setUp(self):
        self.dist = _pack_once()
        self.paths = _paths(self.dist)
        self.zh = _read(self.dist, "README.md")
        self.en = (_read(self.dist, "README.en.md")
                   if "README.en.md" in self.paths else "")

    def test_english_readme_is_shipped(self):
        self.assertIn("README.en.md", self.paths,
                      "README.md 声称有英文版，但打包名单里没有 README.en.md")

    def test_the_two_readmes_link_to_each_other(self):
        self.assertIn("(README.en.md)", self.zh, "中文 README 的语言行没链到英文版")
        self.assertIn("(README.md)", self.en, "英文 README 的语言行没链回中文版")

    def test_language_line_sits_directly_under_the_title(self):
        for name, txt in (("README.md", self.zh), ("README.en.md", self.en)):
            head = txt.splitlines()[:4]
            self.assertTrue(head and head[0].startswith("# "), f"{name} 第一行应是标题")
            self.assertTrue(any("README" in l for l in head[1:4]),
                            f"{name} 标题下 3 行内必须有语言切换行，实际是 {head!r}")

    def test_english_covers_every_section_of_the_chinese_one(self):
        """小节数相等 = 英文版不是缩略宣传页；数字相等 = 实测口径没被改写。"""
        n_zh = len([l for l in self.zh.splitlines() if l.startswith("## ")])
        n_en = len([l for l in self.en.splitlines() if l.startswith("## ")])
        self.assertEqual(n_en, n_zh, f"英文版小节数 {n_en} ≠ 中文版 {n_zh}，两边口径已漂移")

    def test_english_keeps_the_measured_claims(self):
        for token in ("34", "81.8", "96.4", "4469", "40 MB", "127.0.0.1", "exit 1",
                      "kb/baseline.json", "kb_atomic"):
            self.assertIn(token, self.en, f"英文版丢了可核对的实测口径：{token}")

    def test_english_declares_what_is_not_published(self):
        for token in ("corpus", "memory", "baseline"):
            self.assertIn(token, self.en, f"英文版没交代发布边界（缺关键词 {token}）")


if __name__ == "__main__":
    unittest.main(verbosity=2)
