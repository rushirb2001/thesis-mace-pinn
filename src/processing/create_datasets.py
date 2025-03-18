#!/usr/bin/env python3
"""
processfiles.py
Preview → confirm → zip → delete files of a chosen extension
(never deletes itself, even if it matches).

Uses Rich for progress bars.
Install:  pip install rich

Usage examples
--------------
  python processfiles.py                    # current dir, *.py (default)
  python processfiles.py . .txt             # current dir, *.txt
  python processfiles.py /data .csv         # /data dir, *.csv
"""

import os
import sys
import time
import zipfile
from pathlib import Path
from datetime import datetime

try:
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.console import Console
except ImportError:
    sys.exit("❌  The 'rich' library is required. Install it with:  pip install rich")

DELAY = 0.1  # seconds
SCRIPT_PATH = Path(__file__).resolve()
console = Console()

target_dir_arg = "."
ext_arg = ".py"

if len(sys.argv) == 2:
    # One extra argument: treat it as the extension *unless* it's a directory
    if Path(sys.argv[1]).is_dir():
        target_dir_arg = sys.argv[1]
    else:
        ext_arg = sys.argv[1]
elif len(sys.argv) >= 3:
    target_dir_arg = sys.argv[1]
    ext_arg = sys.argv[2]

if not ext_arg.startswith("."):
    ext_arg = "." + ext_arg  # ensure leading dot

target_dir = Path(target_dir_arg).resolve()


def gather_files(target_dir: Path, ext: str):
    return [
        f for f in target_dir.glob(f"*{ext}")
        if f.is_file() and not f.samefile(SCRIPT_PATH)
    ]


def confirm_file_list(files, ext):
    console.print(f"\nFound the following [bold]{ext}[/] files:")
    for f in files:
        console.print(f" • {f.name}")
    resp = console.input(f"\nZip & delete these {len(files)} files? [y/N] ").strip().lower()
    return resp in {"y", "yes"}


def make_archive(target_dir: Path, files, ext):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = target_dir / f"{ext.lstrip('.')}_backup_{ts}.zip"

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(compact=True),
    ) as progress:
        task = progress.add_task("Zipping", total=len(files))
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, arcname=f.name)
                time.sleep(DELAY)
                progress.advance(task)
    return zip_path


def delete_files(files):
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(compact=True),
    ) as progress:
        task = progress.add_task("Deleting", total=len(files))
        for f in files:
            try:
                f.unlink()
            except Exception as e:
                console.print(f"[red]⚠️  Could not delete {f.name}: {e}")
            time.sleep(DELAY)
            progress.advance(task)


def main():
    # positional args: dir [ext]
    if len(sys.argv) >= 2 and not sys.argv[1].startswith('.'):
        target_dir_arg = sys.argv[1]
        ext_arg = sys.argv[2] if len(sys.argv) >= 3 else '.py'
    else:
        target_dir_arg = '.'
        ext_arg = sys.argv[1] if len(sys.argv) >= 2 else '.py'

    target_dir = Path(target_dir_arg).resolve()
    if not target_dir.is_dir():
        sys.exit(f"Error: {target_dir} is not a directory.")
    if not ext_arg.startswith('.'):
        ext_arg = '.' + ext_arg

    os.chdir(target_dir)

    files = gather_files(target_dir, ext_arg)
    if not files:
        console.print(f"No {ext_arg} files found — nothing to do.")
        return

    if not confirm_file_list(files, ext_arg):
        console.print("[yellow]Abort: no files were changed.")
        return

    zip_path = make_archive(target_dir, files, ext_arg)
    delete_files(files)
    console.print(f"\n[green]✅ Done.[/] Archive saved at [bold]{zip_path}[/]")


if __name__ == "__main__":
    main()
