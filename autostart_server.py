# -*- coding: utf-8 -*-
"""龙妖识别器服务守护启动器：
- 若 5000 端口已有服务在跑，直接退出（防重复启动）
- 否则以分离进程方式拉起 flask_server.py，日志写入 logs/flask_task.log
由 Windows 启动文件夹的 run_server.bat 调用。
"""
import os
import socket
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.path.dirname(sys.executable), "python.exe")
if not os.path.exists(PY):  # 兜底: 固定 venv 路径
    PY = r"C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
LOG = os.path.join(BASE, "logs", "flask_task.log")


def port_up():
    s = socket.socket()
    s.settimeout(2)
    try:
        s.connect(("127.0.0.1", 5000))
        return True
    except Exception:
        return False
    finally:
        s.close()


def main():
    if port_up():
        return
    os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)
    flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    logf = open(LOG, "a", encoding="utf-8")
    subprocess.Popen(
        [PY, "-u", os.path.join(BASE, "flask_server.py")],
        cwd=BASE, stdout=logf, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, creationflags=flags,
    )
    logf.close()
    # 等待拉起成功（最多 120 秒，首轮扫描较慢）
    for _ in range(60):
        time.sleep(2)
        if port_up():
            break


if __name__ == "__main__":
    main()
