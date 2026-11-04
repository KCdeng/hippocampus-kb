"""运维加固 A 组的失败测试（个人服务器、单用户、只监听 127.0.0.1）。

三处实测缺口（2026-11-04 生产风险盘点）：
  A① query_log.jsonl 没有轮转。健康探测只在它大到 50MB 时报警，报警之后它还在长——
      每查询一行、每行几百字节，跑一年就是几个 GB 的纯文本，把 NAS 盘填满没人提醒。
  A② _start.ps1 用 Start-Process -RedirectStandardOutput/Error 起服务，PowerShell 以
      CREATE_ALWAYS 打开这两个文件：每次执行启动脚本都会把上一世的现场截成 0 字节。
      更糟的是端口被占用时 kb_serve 自己会 FATAL 退出，但截断在进程创建时就已经发生，
      所以"启动失败"的那一次同样毁掉了正在跑的服务的日志。崩溃前现场拿不回来。
  A③ 重载结果只活在进程内存 RELOAD 里（/stats 可见），进程一死就没了；reload_zone 的
      except 分支把 traceback 丢掉，只剩一行 type+msg，事后查不到是哪一步炸的。
"""
import importlib.util
import json
import os
import socket
import subprocess
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


def _txt(p):
    with open(p, encoding="utf-8") as r:
        return r.read()


def _bytes(p):
    with open(p, "rb") as r:
        return r.read()


class RotateTest(unittest.TestCase):
    """kb_log_archive.rotate：纯函数，级联改名并守住份数。"""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="rot_")
        self.la = _mod(os.path.join(KB, "kb_log_archive.py"), "la_under_test")
        self.log = os.path.join(self.d, "query_log.jsonl")

    def _mk(self, name, body):
        with open(os.path.join(self.d, name), "w", encoding="utf-8") as w:
            w.write(body)
        return os.path.join(self.d, name)

    def test_rotate_shifts_cascade_and_drops_oldest(self):
        self._mk("query_log.jsonl", "cur\n")
        self._mk("query_log.jsonl.1", "one\n")
        self._mk("query_log.jsonl.2", "two\n")
        self._mk("query_log.jsonl.3", "three\n")     # keep=3 → 这一份该被挤掉
        got = self.la.rotate(self.log, keep=3)
        self.assertEqual(got, self.log + ".1")
        self.assertFalse(os.path.exists(self.log), "当前日志没被改名出去")
        self.assertEqual(_txt(self.log + ".1"), "cur\n")
        self.assertEqual(_txt(self.log + ".2"), "one\n")
        self.assertEqual(_txt(self.log + ".3"), "two\n")
        self.assertFalse(os.path.exists(self.log + ".4"), "超出 keep 的最旧档没删，等于无限增长")

    def test_rotate_missing_file_is_noop(self):
        self.assertIsNone(self.la.rotate(self.log, keep=3))

    def test_archive_timestamped_keeps_newest_n(self):
        """服务日志跨重启留档：当前日志带时间戳归档，归档总数受 keep 约束。"""
        for i in range(4):
            self._mk(f"serve.out.log.2026110{i}-000000", f"old{i}\n")
        cur = self._mk("serve.out.log", "live\n")
        got = self.la.archive_timestamped(cur, keep=3)
        self.assertTrue(os.path.exists(got))
        self.assertNotEqual(os.path.basename(got), "serve.out.log", "没改名就直接留在原地")
        self.assertFalse(os.path.exists(cur), "归档后当前文件应已让位给新实例")
        arch = sorted(f for f in os.listdir(self.d) if f.startswith("serve.out.log."))
        self.assertEqual(len(arch), 3, f"归档没受 keep 约束：{arch}")
        self.assertIn(os.path.basename(got), arch, "刚写的那份归档被自己剪掉了")
        self.assertEqual(_txt(got), "live\n")


class ServeLogRotationTest(unittest.TestCase):
    """kb_serve.log_query 必须自己把 query_log 压在阈值内，不靠外部定时任务。"""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="srvlog_")
        self.serve = _mod(os.path.join(KB, "kb_serve.py"), "serve_rot")
        self.serve.LOG = os.path.join(self.d, "query_log.jsonl")
        self.serve.LOG_MAX_BYTES = 4000
        self.serve.LOG_KEEP = 3

    def test_under_threshold_only_appends(self):
        for i in range(10):
            self.serve.log_query({"query": f"q{i}", "kb": "main", "hits": ["a.md"]})
        self.assertTrue(os.path.exists(self.serve.LOG))
        self.assertFalse(os.path.exists(self.serve.LOG + ".1"), "还没到阈值就轮转，日志会被切碎")
        self.assertEqual(len(_txt(self.serve.LOG).splitlines()), 10)

    def test_over_threshold_rotates_and_bounds_disk(self):
        for i in range(300):
            self.serve.log_query({"query": "查询" * 20, "kb": "main",
                                  "hits": ["注塑机/tools/a.md"] * 5})
        self.assertTrue(os.path.exists(self.serve.LOG + ".1"), "超阈值没轮转")
        self.assertLessEqual(os.path.getsize(self.serve.LOG), self.serve.LOG_MAX_BYTES * 2)
        left = [f for f in os.listdir(self.d) if f.startswith("query_log.jsonl")]
        self.assertLessEqual(len(left), 1 + self.serve.LOG_KEEP, f"轮转份数失控：{sorted(left)}")


class ReloadHistoryTest(unittest.TestCase):
    """重载结果落盘：成功与失败都算，失败要带 traceback。"""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="relh_")
        self.serve = _mod(os.path.join(KB, "kb_serve.py"), "serve_relh")
        self.serve.RELOAD_HISTORY = os.path.join(self.d, "reload_history.jsonl")

    def _rows(self):
        if not os.path.exists(self.serve.RELOAD_HISTORY):
            return []
        with open(self.serve.RELOAD_HISTORY, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def test_record_reload_appends_one_json_line(self):
        self.serve.record_reload("main", True, "main 重载完成")
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["zone"], "main")
        self.assertTrue(rows[0]["ok"])
        self.assertIn("ts", rows[0])

    def test_reload_and_record_persists_failure_with_traceback(self):
        def boom(_embedder):
            raise ValueError("模拟换档失败")
        orig_build, orig_main = self.serve._build_main, self.serve.STATE.get("main")
        self.serve._build_main = boom
        try:
            ok, msg = self.serve.reload_and_record("main")
        finally:
            self.serve._build_main = orig_build
            self.serve.STATE["main"] = orig_main
        self.assertFalse(ok)
        rows = self._rows()
        self.assertEqual(len(rows), 1, "失败没落盘——重启后查不到这次坏档")
        self.assertFalse(rows[0]["ok"])
        self.assertIn("ValueError", rows[0]["message"])
        self.assertIn("模拟换档失败", rows[0].get("traceback") or "",
                      "traceback 丢了：只剩异常名，事后定位不到哪一步")

    def test_success_is_also_recorded(self):
        orig = self.serve.reload_zone
        self.serve.reload_zone = lambda z: (True, f"{z} 重载完成", None)
        try:
            self.serve.reload_and_record("mem")
        finally:
            self.serve.reload_zone = orig
        rows = self._rows()
        self.assertEqual((rows[-1]["zone"], rows[-1]["ok"]), ("mem", True))

    def test_repeated_identical_failure_recorded_once(self):
        """watcher 对失败的区每 10s 重试（基线不更新），同一句话每次追加一条 + 一份 traceback：
        一天就是 8640 条，这份档本身会变成新的增长源。所以连续同区同文的失败只记一次。"""
        for _ in range(5):
            self.serve.reload_and_record("nope")      # 未知区：每次都失败、每次都同一句
        rows = self._rows()
        self.assertEqual(len(rows), 1, f"重复失败被记了 {len(rows)} 遍")
        self.assertFalse(rows[0]["ok"])

    def test_recovered_then_failing_again_is_recorded_again(self):
        self.serve.reload_and_record("nope")
        orig = self.serve.reload_zone
        self.serve.reload_zone = lambda z: (True, f"{z} 重载完成", None)
        try:
            self.serve.reload_and_record("nope")
        finally:
            self.serve.reload_zone = orig
        self.serve.reload_and_record("nope")
        rows = self._rows()
        self.assertEqual([r["ok"] for r in rows], [False, True, False],
                         "中间好转过，下一次失败必须重新记账")

    def test_repeated_failure_prints_stderr_once(self):
        """serve.err.log 现在只归档、不再被截断，而 watcher 每 10s 重试同一个坏区——
        打印不去重的话这份日志一天要写 8640 段 traceback，等于把增长源从
        reload_history.jsonl 搬到 err.log。落盘和打印必须用同一个去重判据。"""
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            for _ in range(5):
                self.serve.reload_and_record("nope")
        n = buf.getvalue().count("RELOAD FAILED")
        self.assertEqual(n, 1, f"同一句失败往 err.log 打了 {n} 遍")

    def test_reload_history_rotates(self):
        """落盘件自己也要有占盘上限，否则只是把增长源从 query_log 换成 reload_history。"""
        self.serve.RELOAD_HIST_MAX_BYTES = 600
        self.serve.RELOAD_HIST_KEEP = 3
        for i in range(60):
            self.serve.record_reload(f"z{i}", False, f"第{i}次不同的失败", "x" * 80)
        arch = [f for f in os.listdir(self.d) if f.startswith("reload_history.jsonl.")]
        self.assertTrue(arch, "reload_history 超阈值没轮转")
        self.assertLessEqual(len(arch), self.serve.RELOAD_HIST_KEEP, f"轮转份数失控：{sorted(arch)}")
        self.assertLessEqual(os.path.getsize(self.serve.RELOAD_HISTORY),
                             self.serve.RELOAD_HIST_MAX_BYTES * 2)


class StartPsLiveLogTest(unittest.TestCase):
    """防截断只能在**服务真在跑**的时候验，所以这条用真实 8310 与真实 serve.out.log。

    断言两件事：脚本以非零码退出（明确告诉你"没启动，因为已经有实例"），
    并且两个日志文件一个字节都没动——包括不许截断。
    """

    PORT = 8310

    def _alive(self):
        s = socket.socket()
        s.settimeout(0.5)
        try:
            return s.connect_ex(("127.0.0.1", self.PORT)) == 0
        finally:
            s.close()

    def test_start_script_leaves_live_service_alone(self):
        if not self._alive():
            self.skipTest("8310 没有在跑：这条测试验的是别把在跑服务的日志截掉，需要活服务")
        out = os.path.join(KB, "serve.out.log")
        err = os.path.join(KB, "serve.err.log")
        before = {p: _bytes(p) for p in (out, err) if os.path.exists(p)}
        scratch = os.path.join(tempfile.gettempdir(), "startps_live.out")
        with open(scratch, "wb") as fh:
            # 别用 capture_output：脚本里 Start-Process 起的后台服务会继承管道，
            # 调用方等 EOF 会永久挂住（agents.md 第 1 条"调用坑"）。重定向到文件。
            rc = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                                 "-File", os.path.join(KB, "_start.ps1")],
                                stdout=fh, stderr=subprocess.STDOUT, timeout=120).returncode
        self.assertNotEqual(rc, 0, "已有实例在跑时 _start.ps1 应非零退出（现在它静默报告启动成功）")
        for p, data in before.items():
            self.assertEqual(_bytes(p), data,
                             f"{os.path.basename(p)} 被动过——启动脚本会截断在跑服务的日志")


if __name__ == "__main__":
    unittest.main(verbosity=2)
