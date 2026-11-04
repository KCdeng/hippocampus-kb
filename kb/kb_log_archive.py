"""kb_log_archive.py — 日志轮转与跨重启留档（纯 stdlib，不依赖检索引擎）。

两类日志要两种留法，别混：
  query_log.jsonl  同一世的追加流水。超过阈值就级联改名（.1/.2/.3），只保最近 keep 份，
                   磁盘上限因此可算（阈值 ×(keep+1)）。
  serve.out.log / serve.err.log  跨世的进程现场。Start-Process 的 -RedirectStandardOutput
                   以 CREATE_ALWAYS 打开它们，进程创建那一刻就把上一世截成 0 字节——实测
                   端口被占、新实例自己 FATAL 退出时也一样截（2026-11-04：在跑服务的 120 字节
                   启动日志被死实例的一行 FATAL 顶掉）。所以启动脚本必须先把旧档带时间戳挪走。

命令行给 _start.ps1 用，也给手工清盘用：
    python kb_log_archive.py archive kb/serve.out.log --keep 3
    python kb_log_archive.py rotate  kb/query_log.jsonl --keep 3
"""
import argparse
import datetime
import glob
import os


def _list_archives(path):
    """path 的所有归档，按名字排序。归档名带 YYYYmmdd-HHMMSS，名字序即时间序。"""
    return sorted(glob.glob(path + ".*"))


def rotate(path, keep=3):
    """当前日志改名为 .1，已有的 .1/.2… 依次后移，第 keep 份挤掉。

    返回新 .1 的路径；当前文件不存在就什么都不做，返回 None。
    调用方负责并发（kb_serve 里轮转与追加同在 _log_lock 内，不会把写句柄踩空）。"""
    if not os.path.exists(path):
        return None
    oldest = f"{path}.{keep}"
    if os.path.exists(oldest):
        os.remove(oldest)
    for i in range(keep - 1, 0, -1):
        src = f"{path}.{i}"
        if os.path.exists(src):
            os.replace(src, f"{path}.{i + 1}")
    os.replace(path, f"{path}.1")
    return f"{path}.1"


def archive_timestamped(path, keep=3):
    """把当前文件带时间戳归档（用于跨重启留档），归档总数压在 keep 份内。

    返回归档路径；当前文件不存在返回 None。"""
    if not os.path.exists(path):
        return None
    dst = f"{path}.{datetime.datetime.now():%Y%m%d-%H%M%S}"
    os.replace(path, dst)
    for stale in _list_archives(path)[:-keep]:
        try:
            os.remove(stale)
        except OSError:
            pass        # 归档剪不掉不该拦住服务启动
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rotate", "archive"])
    ap.add_argument("path")
    ap.add_argument("--keep", type=int, default=3)
    a = ap.parse_args()
    got = rotate(a.path, a.keep) if a.cmd == "rotate" else archive_timestamped(a.path, a.keep)
    print(got or "noop（文件不存在）")


if __name__ == "__main__":
    main()
