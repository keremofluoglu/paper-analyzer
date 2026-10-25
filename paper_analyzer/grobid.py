"""Start GROBID (Docker inside a WSL distro) for the duration of an ingest, then stop it to free RAM.
Anything missing (no WSL, no distro, no Docker) just means ingest falls back to PyMuPDF."""
import os
import subprocess
import sys
import time
from contextlib import contextmanager

from . import config, parse


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _wsl(*args: str, timeout: int = 60) -> subprocess.CompletedProcess | None:
    env = {**os.environ, "WSL_UTF8": "1"}
    try:
        return subprocess.run(["wsl", *args], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _distro_running() -> bool:
    r = _wsl("--list", "--running", "--quiet")
    return bool(r and config.GROBID_WSL_DISTRO in r.stdout.split())


def _docker(*args: str, timeout: int = 60) -> subprocess.CompletedProcess | None:
    return _wsl("-d", config.GROBID_WSL_DISTRO, "-u", "root", "--", "docker", *args, timeout=timeout)


_proc: subprocess.Popen | None = None


def start(wait: int = 180) -> bool:
    """Start the GROBID container and wait until it answers. Returns True when GROBID is usable.
    The container runs attached to a wsl.exe process: WSL shuts an idle distro down as soon as
    no wsl.exe client is left, so a detached container would die with it."""
    global _proc
    _docker("rm", "-f", config.GROBID_CONTAINER)
    try:
        _proc = subprocess.Popen(
            ["wsl", "-d", config.GROBID_WSL_DISTRO, "-u", "root", "--", "docker", "run", "--rm",
             "--name", config.GROBID_CONTAINER, "-p", "8070:8070", "--init", config.GROBID_IMAGE],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    except OSError as e:
        log(f"  GROBID not started ({e}) - using PyMuPDF")
        return False
    log("  starting GROBID...")
    end = time.time() + wait
    while time.time() < end:
        if _proc.poll() is not None:
            err = _proc.stderr.read().decode("utf-8", "replace").replace("\x00", "").strip().splitlines()
            log(f"  GROBID not started ({err[-1] if err else 'exit ' + str(_proc.returncode)}) - using PyMuPDF")
            _proc = None
            return False
        if parse.grobid_alive():
            return True
        time.sleep(3)
    log("  GROBID did not answer in time - using PyMuPDF")
    stop(shutdown_distro=False)
    return False


def stop(shutdown_distro: bool):
    global _proc
    _docker("stop", config.GROBID_CONTAINER)
    if _proc is not None:
        try:
            _proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _proc.kill()
        _proc = None
    if shutdown_distro:
        _wsl("--terminate", config.GROBID_WSL_DISTRO)


@contextmanager
def session():
    """Use a running GROBID if there is one; otherwise start it here and stop it on exit."""
    if parse.grobid_alive() or not config.GROBID_AUTO:
        yield
        return
    was_running = _distro_running()
    started = start()
    parse.reset_grobid()
    try:
        yield
    finally:
        if started:
            stop(shutdown_distro=not was_running)
            log("  GROBID stopped")
        elif not was_running:
            _wsl("--terminate", config.GROBID_WSL_DISTRO)
        parse.reset_grobid()
