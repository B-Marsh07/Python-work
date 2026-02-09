#!/usr/bin/env python3
"""Automaton: a playful CLI for lightweight system housekeeping."""

from __future__ import annotations

import datetime as dt
import os
import platform
import shutil
import subprocess
import sys
from typing import List, Tuple

BANNER = r"""
    ⚡ Automaton: Pocket Ops Console ⚡
    ---------------------------------
    Tiny automations. Big vibes.
"""

MENU_ITEMS = [
    (
        "1",
        "System snapshot",
        "Show memory-hungry processes and temp folder size.",
    ),
    (
        "2",
        "Clean temp files",
        "Free up space by clearing OS temp files (with confirmation).",
    ),
    (
        "3",
        "Find & close idle apps",
        "List apps idle for 45+ minutes and optionally close them.",
    ),
    (
        "4",
        "Launch an app/command",
        "Start a program by typing its command.",
    ),
    (
        "5",
        "Terminate by PID",
        "Close a specific process by PID.",
    ),
    (
        "6",
        "Lower priority",
        "Reduce CPU priority of a process (gentle optimization).",
    ),
    ("Q", "Quit", "Exit Automaton."),
]

IDLE_MINUTES_DEFAULT = 45


def run_command(cmd: List[str]) -> Tuple[int, str, str]:
    """Run a command and capture output."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def human_bytes(num_bytes: float) -> str:
    step = 1024.0
    unit = "B"
    for next_unit in ["KB", "MB", "GB", "TB"]:
        if num_bytes < step:
            break
        num_bytes /= step
        unit = next_unit
    return f"{num_bytes:.1f} {unit}"


def temp_dir() -> str:
    return os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp"


def folder_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def top_memory_processes(limit: int = 5) -> List[Tuple[str, str, str]]:
    system = platform.system().lower()
    if system == "windows":
        cmd = ["tasklist", "/FO", "CSV", "/NH"]
        code, out, err = run_command(cmd)
        if code != 0:
            return [("-", "-", f"tasklist failed: {err}")]
        rows = []
        for line in out.splitlines():
            parts = [part.strip('"') for part in line.split("\",")]
            if len(parts) < 5:
                continue
            name, pid, _, mem, _ = parts[:5]
            rows.append((pid, name, mem))
        return rows[:limit]

    cmd = ["ps", "-eo", "pid,comm,rss", "--sort=-rss"]
    code, out, err = run_command(cmd)
    if code != 0:
        return [("-", "-", f"ps failed: {err}")]
    rows = []
    for line in out.splitlines()[1 : limit + 1]:
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            continue
        pid, comm, rss = parts
        rows.append((pid, comm, human_bytes(float(rss) * 1024)))
    return rows


def print_snapshot() -> None:
    print("\n📊 System Snapshot")
    temp_path = temp_dir()
    size = folder_size(temp_path)
    print(f"Temp path: {temp_path}")
    print(f"Temp size: {human_bytes(size)}")
    print("\nTop memory processes:")
    for pid, name, mem in top_memory_processes():
        print(f"  PID {pid:>6} | {name:<25} | {mem}")


def confirm(prompt: str) -> bool:
    choice = input(f"{prompt} [y/N]: ").strip().lower()
    return choice in {"y", "yes"}


def clean_temp() -> None:
    path = temp_dir()
    print(f"\n🧹 Clean Temp Files")
    size_before = folder_size(path)
    print(f"Temp folder: {path}")
    print(f"Current size: {human_bytes(size_before)}")
    if not confirm("Proceed with cleanup?"):
        print("Cancelled.")
        return
    removed = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                os.remove(os.path.join(root, name))
                removed += 1
            except OSError:
                continue
    size_after = folder_size(path)
    print(f"Removed files: {removed}")
    print(f"New size: {human_bytes(size_after)}")


def parse_ps_time(value: str) -> int:
    """Parse ps TIME output into seconds."""
    parts = value.split("-")
    if len(parts) == 2:
        days = int(parts[0])
        time_part = parts[1]
    else:
        days = 0
        time_part = parts[0]
    fields = [int(p) for p in time_part.split(":")]
    while len(fields) < 3:
        fields.insert(0, 0)
    hours, minutes, seconds = fields
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def idle_candidates(minutes: int) -> List[Tuple[str, str, str]]:
    """Approximate idle processes by comparing elapsed time and CPU time."""
    system = platform.system().lower()
    threshold_seconds = minutes * 60
    if system == "windows":
        cmd = ["powershell", "-Command", "Get-Process | Select-Object Id,ProcessName,CPU,StartTime"]
        code, out, err = run_command(cmd)
        if code != 0:
            return [("-", "-", f"powershell failed: {err}")]
        lines = [line for line in out.splitlines() if line.strip()]
        candidates = []
        now = dt.datetime.now()
        for line in lines[2:]:
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            pid, name, cpu_str, start_str = parts
            try:
                start_time = dt.datetime.fromisoformat(start_str)
            except ValueError:
                continue
            elapsed = (now - start_time).total_seconds()
            cpu_seconds = float(cpu_str) if cpu_str not in {"", "-"} else 0.0
            if elapsed - cpu_seconds >= threshold_seconds:
                candidates.append((pid, name, f"idle ~{int(elapsed // 60)}m"))
        return candidates

    cmd = ["ps", "-eo", "pid,comm,etime,time"]
    code, out, err = run_command(cmd)
    if code != 0:
        return [("-", "-", f"ps failed: {err}")]
    candidates = []
    for line in out.splitlines()[1:]:
        parts = line.split(maxsplit=3)
        if len(parts) != 4:
            continue
        pid, comm, etime, cpu_time = parts
        try:
            elapsed = parse_ps_time(etime)
            cpu_seconds = parse_ps_time(cpu_time)
        except ValueError:
            continue
        if elapsed - cpu_seconds >= threshold_seconds:
            candidates.append((pid, comm, f"idle ~{elapsed // 60}m"))
    return candidates


def close_idle_apps() -> None:
    print("\n🛌 Idle App Finder")
    minutes = input(f"Idle minutes threshold [{IDLE_MINUTES_DEFAULT}]: ").strip()
    threshold = int(minutes) if minutes else IDLE_MINUTES_DEFAULT
    candidates = idle_candidates(threshold)
    if not candidates:
        print("No idle apps detected.")
        return
    print("Possible idle apps:")
    for pid, name, info in candidates:
        print(f"  PID {pid:>6} | {name:<25} | {info}")
    if confirm("Terminate all listed processes?"):
        for pid, name, _ in candidates:
            terminate_process(pid, name)


def launch_app() -> None:
    print("\n🚀 Launch App")
    cmd = input("Enter command (e.g., 'code', 'notepad', '/usr/bin/top'): ").strip()
    if not cmd:
        print("No command provided.")
        return
    try:
        subprocess.Popen(cmd, shell=True)
        print("Launched.")
    except OSError as exc:
        print(f"Failed to launch: {exc}")


def terminate_process(pid: str, name: str | None = None) -> None:
    label = f"{name} (PID {pid})" if name else f"PID {pid}"
    system = platform.system().lower()
    if system == "windows":
        cmd = ["taskkill", "/PID", pid, "/F"]
    else:
        cmd = ["kill", "-9", pid]
    code, _, err = run_command(cmd)
    if code == 0:
        print(f"Closed {label}.")
    else:
        print(f"Could not close {label}: {err}")


def kill_by_pid() -> None:
    print("\n🧨 Terminate by PID")
    pid = input("PID to terminate: ").strip()
    if not pid:
        print("No PID provided.")
        return
    if confirm(f"Terminate PID {pid}?"):
        terminate_process(pid)


def lower_priority() -> None:
    print("\n🌙 Lower Priority")
    pid = input("PID to deprioritize: ").strip()
    if not pid:
        print("No PID provided.")
        return
    system = platform.system().lower()
    if system == "windows":
        cmd = [
            "powershell",
            "-Command",
            f"$p=Get-Process -Id {pid}; $p.PriorityClass='BelowNormal'",
        ]
    else:
        cmd = ["renice", "10", "-p", pid]
    code, _, err = run_command(cmd)
    if code == 0:
        print("Priority lowered.")
    else:
        print(f"Could not change priority: {err}")


def print_menu() -> None:
    print(BANNER)
    for key, title, desc in MENU_ITEMS:
        print(f"[{key}] {title} - {desc}")


def main() -> None:
    print_menu()
    while True:
        choice = input("\nSelect option: ").strip().upper()
        if choice == "1":
            print_snapshot()
        elif choice == "2":
            clean_temp()
        elif choice == "3":
            close_idle_apps()
        elif choice == "4":
            launch_app()
        elif choice == "5":
            kill_by_pid()
        elif choice == "6":
            lower_priority()
        elif choice in {"Q", "QUIT", "EXIT"}:
            print("Stay optimized. ✨")
            break
        else:
            print("Unknown choice. Try again.")


if __name__ == "__main__":
    main()
