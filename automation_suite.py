import gc
import getpass
import importlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime

APP_NAME = "PulseKeep Automator"
LOG_FILE = "pulsekeep.log"
DEFAULT_IDLE_MINUTES = 45
TEMP_CLEANUP_HOURS = 24


def log_event(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def supports_color():
    return sys.stdout.isatty()


def colorize(text, code):
    if not supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def banner():
    title = colorize(APP_NAME, "96")
    subtitle = colorize("Keep your system calm, clean, and focused.", "90")
    print(
        colorize(
            """
   ___        _          _  __
  / _ \ _   _| | ___ ___| |/ /___  ___ _ __
 | | | | | | | |/ _ / __| ' // _ \\/ _ \\ '__|
 | |_| | |_| | |  __\__ \ . \  __/  __/ |
  \__\_\\__,_|_|\___|___/_|\_\\___|\___|_|
""",
            "95",
        )
    )
    print(title)
    print(subtitle)


def get_psutil():
    spec = importlib.util.find_spec("psutil")
    if spec is None:
        return None
    return importlib.import_module("psutil")


def human_bytes(num):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num)
    for unit in units:
        if size < 1024:
            return f"{size:,.1f} {unit}"
        size /= 1024
    return f"{size:,.1f} PB"


def system_pulse():
    psutil = get_psutil()
    print("\n-- System Pulse --")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"User: {getpass.getuser()}")

    if psutil:
        memory = psutil.virtual_memory()
        print(
            f"Memory: {human_bytes(memory.used)} used / {human_bytes(memory.total)} total"
        )
        print(f"CPU usage: {psutil.cpu_percent(interval=0.2)}%")
    else:
        print("Memory: psutil not available (install to unlock live stats)")
    disk = shutil.disk_usage(os.path.expanduser("~"))
    print(f"Home disk: {human_bytes(disk.used)} used / {human_bytes(disk.total)} total")
    log_event("System pulse checked.")


def free_memory():
    print("\n-- Memory Calm --")
    collected = gc.collect()
    details = [f"Garbage collection pass complete. Objects collected: {collected}."]

    if platform.system() == "Linux":
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            result = libc.malloc_trim(0)
            details.append(
                "Trimmed Python heap back to the OS." if result == 1 else "Heap trim skipped."
            )
        except OSError:
            details.append("Heap trim not available on this system.")
    elif platform.system() == "Windows":
        details.append("Tip: closing heavy apps can further reduce memory use.")
    else:
        details.append("Memory trim is platform-specific; GC completed.")

    print("\n".join(details))
    log_event("Memory cleanup executed.")


def cleanup_temp(dry_run=True):
    print("\n-- Temp Vault Sweep --")
    cutoff = time.time() - (TEMP_CLEANUP_HOURS * 3600)
    temp_dir = tempfile.gettempdir()
    reclaimed = 0
    scanned = 0

    for root, _, files in os.walk(temp_dir):
        for filename in files:
            path = os.path.join(root, filename)
            try:
                stats = os.stat(path)
            except OSError:
                continue
            scanned += 1
            if stats.st_mtime < cutoff:
                reclaimed += stats.st_size
                if not dry_run:
                    try:
                        os.remove(path)
                    except OSError:
                        continue

    mode = "Dry run" if dry_run else "Cleanup"
    print(f"{mode}: scanned {scanned} files in {temp_dir}")
    print(f"Potential space to reclaim: {human_bytes(reclaimed)}")
    log_event(f"Temp sweep ({mode.lower()}) completed. Reclaimable={reclaimed} bytes.")


def _collect_idle_processes(min_idle_minutes):
    psutil = get_psutil()
    if not psutil:
        print("psutil not installed. Run: pip install psutil to manage apps.")
        return []

    threshold = time.time() - (min_idle_minutes * 60)
    user = getpass.getuser()
    processes = []

    candidates = list(psutil.process_iter(["pid", "name", "create_time", "username"]))
    for proc in candidates:
        try:
            proc.cpu_percent(None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    time.sleep(0.1)

    for proc in candidates:
        try:
            info = proc.info
            if not info.get("create_time") or info.get("username") != user:
                continue
            if info["create_time"] > threshold:
                continue
            cpu_usage = proc.cpu_percent(None)
            if cpu_usage == 0.0:
                processes.append(
                    {
                        "pid": info["pid"],
                        "name": info.get("name") or "unknown",
                        "age_minutes": int((time.time() - info["create_time"]) / 60),
                        "cpu": cpu_usage,
                    }
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return processes


def list_idle_apps():
    print("\n-- Idle App Radar --")
    idle = _collect_idle_processes(DEFAULT_IDLE_MINUTES)
    if not idle:
        print("No idle apps detected beyond the threshold.")
        return
    print(
        f"Idle apps (>{DEFAULT_IDLE_MINUTES} minutes, zero CPU in last check):"
    )
    for idx, proc in enumerate(idle, start=1):
        print(
            f"[{idx}] {proc['name']} (PID {proc['pid']}) - {proc['age_minutes']} min old"
        )
    log_event(f"Idle app list generated ({len(idle)} items).")


def close_idle_apps():
    print("\n-- Focus Mode: Close Idle Apps --")
    idle = _collect_idle_processes(DEFAULT_IDLE_MINUTES)
    if not idle:
        print("No idle apps to close.")
        return

    for idx, proc in enumerate(idle, start=1):
        print(f"[{idx}] {proc['name']} (PID {proc['pid']})")

    choice = input("Close all idle apps? (y/N): ").strip().lower()
    if choice != "y":
        print("Canceled.")
        return

    psutil = get_psutil()
    closed = 0
    for proc in idle:
        try:
            psutil.Process(proc["pid"]).terminate()
            closed += 1
        except psutil.Error:
            continue

    print(f"Focus mode complete. Closed {closed} app(s).")
    log_event(f"Closed idle apps: {closed}.")


def quick_launch():
    print("\n-- Quick Launch Deck --")
    options = {
        "1": ("Open home folder", os.path.expanduser("~")),
        "2": ("Open temp folder", tempfile.gettempdir()),
        "3": ("Open log file", os.path.abspath(LOG_FILE)),
    }
    for key, (label, _) in options.items():
        print(f"[{key}] {label}")
    selection = input("Pick a destination: ").strip()
    target = options.get(selection)
    if not target:
        print("Invalid selection.")
        return

    path = target[1]
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)

    print(f"Opened: {path}")
    log_event(f"Quick launch opened {path}.")


def show_menu():
    menu_items = [
        (
            "1",
            "System Pulse",
            "Quick snapshot of memory, CPU, and disk health.",
        ),
        (
            "2",
            "Memory Calm",
            "Run garbage collection and trim memory where possible.",
        ),
        (
            "3",
            "Temp Vault Sweep",
            "Estimate cleanup space in the temp directory (24h+ files).",
        ),
        (
            "4",
            "Idle App Radar",
            f"List apps idle for more than {DEFAULT_IDLE_MINUTES} minutes.",
        ),
        (
            "5",
            "Focus Mode",
            f"Close idle apps older than {DEFAULT_IDLE_MINUTES} minutes.",
        ),
        (
            "6",
            "Quick Launch Deck",
            "Open handy folders and logs.",
        ),
        ("7", "Exit", "Leave PulseKeep calmly."),
    ]

    print("\n-- Command Menu --")
    for key, title, desc in menu_items:
        print(f"[{key}] {title:<20} - {desc}")


def main():
    os.system("cls" if os.name == "nt" else "clear")
    banner()
    log_event("PulseKeep started.")

    actions = {
        "1": system_pulse,
        "2": free_memory,
        "3": lambda: cleanup_temp(dry_run=True),
        "4": list_idle_apps,
        "5": close_idle_apps,
        "6": quick_launch,
    }

    while True:
        show_menu()
        choice = input("Select an option [1-7]: ").strip()
        if choice == "7":
            log_event("PulseKeep exited.")
            print("Stay light. Goodbye.")
            break
        action = actions.get(choice)
        if not action:
            print("Invalid option. Try again.")
            continue
        action()
        input("\nPress Enter to return to the menu...")


if __name__ == "__main__":
    main()
