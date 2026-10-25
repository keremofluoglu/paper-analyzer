"""Single entry point for the paper tool. Works from any working directory:
    "<this folder>\\.venv\\Scripts\\python.exe" "<this folder>\\paper.py" <command> [options]
See AI_INSTRUCTIONS.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8")  # Windows consoles default to a legacy code page

from paper_analyzer.cli import app  # noqa: E402

app()
