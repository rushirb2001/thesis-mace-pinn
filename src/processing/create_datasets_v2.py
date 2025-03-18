#!/usr/bin/env python3
"""
processfiles.py ─ interactively choose folders to zip, then delete them.

Features
--------
• If you give folder paths as arguments → operates directly on them.
• If you give *no* folder arguments     → lists first-level sub-dirs of $PWD,
  lets you choose with a range syntax like 1,3-5,8.
• Shows your selection twice (immediate preview + final confirmation).
• Creates a single ZIP archive preserving relative paths.
• Uses Rich progress bars (install with:  pip install rich).
• Inserts a 0.1 s delay per step for smooth animation.
• Safety: aborts if this script resides inside any target folder.

Examples
--------
# Interactive menu (recommended when many sub-folders):
python processfiles.py

# Non-interactive:
python processfiles.py results figures /data/old_runs
"""

from __future__ import annotations
import os, sys, time, zipfile, shutil, re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
except ImportError:
    sys.exit("❌  Requires the 'rich' library.  Install with:  pip install rich")

DELAY = 0.1
SCRIPT = Path(__file__).resolve()
console = Console()

# ───────────────────────── helpers ────────────────────────────
def first_level_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])

def parse_selection(sel: str, max_n: int) -> List[int]:
    """Convert '1,3-5,8' to [1,3,4,5,8] (1-based indices)."""
    out = set()
    for part in sel.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return [i for i in sorted(out) if 1 <= i <= max_n]

def confirm(folders: List[Path]) -> bool:
    console.print("\nThe following folders will be zipped and then deleted:")
    for f in folders:
        console.print(f" • [bold]{f}[/]")
    return console.input("\nProceed? [y/N] ").strip().lower() in {"y", "yes"}

def gather_files(folders: List[Path]) -> List[Tuple[Path, str]]:
    files: List[Tuple[Path, str]] = []
    for folder in folders:
        for fp in folder.rglob("*"):
            if fp.is_file():
                files.append((fp, str(fp.relative_to(folder.parent))))
    return files

def zip_files(files: List[Tuple[Path, str]]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zpath = Path.cwd() / f"folders_backup_{ts}.zip"
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(compact=True),
    ) as prog, zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        task = prog.add_task("Zipping", total=len(files))
        for src, arc in files:
            zf.write(src, arcname=arc)
            time.sleep(DELAY)
            prog.advance(task)
    return zpath

def delete_folders(folders: List[Path]):
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%"
    ) as prog:
        task = prog.add_task("Deleting", total=len(folders))
        for folder in folders:
            try:
                shutil.rmtree(folder)
            except Exception as e:
                console.print(f"[red]⚠️  Could not delete {folder}: {e}")
            time.sleep(DELAY)
            prog.advance(task)

def safety_check(folders: List[Path]):
    for p in folders:
        if SCRIPT.is_relative_to(p):
            sys.exit(f"Safety abort: this script is located inside '{p}'.")

# ───────────────────────── main ───────────────────────────────
def main():
    root = Path.cwd()
    argv_raw = sys.argv[1:]

    # If no args → interactive menu
    if not argv_raw:
        subdirs = first_level_dirs(root)
        if not subdirs:
            console.print("[yellow]No sub-directories found to select.[/]")
            sys.exit(0)

        console.print("\nSelect folders to zip/delete (e.g. 1,3-5):")
        for i, d in enumerate(subdirs, 1):
            console.print(f"[cyan]{i:>2}.[/] {d.name}")
        sel_raw = console.input("\nYour choice (blank to abort): ").strip()
        if not sel_raw:
            console.print("[yellow]Abort: no folders chosen.[/]")
            sys.exit(0)

        try:
            indices = parse_selection(sel_raw, len(subdirs))
        except Exception:
            sys.exit("Invalid selection syntax (use commas and dashes).")

        chosen = [subdirs[i - 1] for i in indices]

        # ── immediate double-check preview
        console.print("\nYou selected:")
        for p in chosen:
            console.print(f"  • [bold green]{p.resolve()}[/]")

        folders = chosen
    else:
        # Args provided → treat each as folder path
        folders = [Path(a).expanduser().resolve() for a in argv_raw]

    # Validate / sanitize list
    folders = [p for p in folders if p.exists() and p.is_dir()]
    if not folders:
        console.print("[yellow]No valid folders supplied.[/]")
        sys.exit(0)

    safety_check(folders)
    if not confirm(folders):
        console.print("[yellow]Abort: no changes made.[/]")
        sys.exit(0)

    files = gather_files(folders)
    if not files:
        console.print("[yellow]No files found in chosen folders.[/]")
        sys.exit(0)

    archive = zip_files(files)
    delete_folders(folders)
    console.print(f"\n[green]✅ Done.[/] Archive saved at [bold]{archive}[/]")

if __name__ == "__main__":
    main()
