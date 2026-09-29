"""Auto Git Sync Watcher

Continuously monitors the repository for changes and pushes updates to GitHub.
Runs safely in the background, strictly respecting .gitignore and ensuring
README.md, datasets, and internal scratchpads are never pushed.

Usage:
    python scripts/auto_git_sync.py [--interval 30] [--once]
"""

import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Whitelisted directories and files allowed to be staged and committed
ALLOWED_PATHS = [
    "src",
    "tests",
    "models",
    "docs",
    "scripts",
    "data/download_davis.py",
    "pyproject.toml",
    ".gitignore",
    ".python-version",
    "uv.lock",
]


def run_git(cmd: list[str]) -> tuple[int, str, str]:
    """Execute a git command in the repository root."""
    try:
        proc = subprocess.run(
            ["git"] + cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def sync_cycle() -> bool:
    """Run a single check, stage, commit, and push cycle.
    
    Returns True if changes were committed and pushed, False otherwise.
    """
    code, status_out, _ = run_git(["status", "--porcelain"])
    if code != 0 or not status_out:
        return False

    # Stage explicitly allowed paths to guarantee no leakage
    for path in ALLOWED_PATHS:
        full_path = REPO_ROOT / path
        if full_path.exists():
            run_git(["add", path])

    # Check if there is anything actually staged
    code, _, _ = run_git(["diff", "--cached", "--quiet"])
    if code == 0:
        # Nothing staged
        return False

    # Get short summary of staged files
    _, stat_summary, _ = run_git(["diff", "--cached", "--stat"])
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_msg = f"chore(sync): automated sync at {now_str}\n\n{stat_summary}"

    # Commit
    code, commit_out, commit_err = run_git(["commit", "-m", commit_msg])
    if code != 0:
        print(f"[{now_str}] Commit failed: {commit_err or commit_out}", flush=True)
        return False

    print(f"[{now_str}] Committed staged changes:\n{commit_out}", flush=True)

    # Push to origin main
    code, push_out, push_err = run_git(["push", "origin", "main"])
    if code != 0:
        print(f"[{now_str}] Push warning/error: {push_err or push_out}", flush=True)
        return False

    print(f"[{now_str}] Successfully pushed to GitHub origin/main.\n{push_out}", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto Git Sync Watcher")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds (default: 30)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    print(f"[INFO] Auto Git Sync Watcher active for {REPO_ROOT}", flush=True)
    print(f"[INFO] Target remote: origin/main | Interval: {args.interval}s", flush=True)
    print(f"[INFO] Strict whitelist enforced: {', '.join(ALLOWED_PATHS)}", flush=True)

    if args.once:
        synced = sync_cycle()
        if not synced:
            print("[INFO] Working tree clean. Nothing to sync.", flush=True)
        return

    while True:
        try:
            sync_cycle()
        except Exception as e:
            print(f"[ERROR] Sync loop exception: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
