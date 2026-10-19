#!/usr/bin/env python3
"""Insert the bench-2 section (templates/readme-b2.{en,ja}.md + tables) at the top of README.md / README.ja.md.

  python scripts/render_readme_b2.py

The section goes right under the language switch; what was there stays below under an
"earlier runs" heading. Running it again replaces the section.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = "results/2026-10-m1max-64gb-b2"
START, END = "<!-- b2:start -->", "<!-- b2:end -->"
FILES = {"en": ("README.md", "English | [日本語](README.ja.md)", "## Earlier runs: thinking off (2026-09 to 2026-10-14)"),
         "ja": ("README.ja.md", "[English](README.md) | 日本語", "## 以前の計測: thinking なし（2026-09〜2026-10-14）")}


def tables(model: str, lang: str) -> str:
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "summarize_b2.py"), RUN, "--readme", model, lang],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


for lang, (name, switch, earlier) in FILES.items():
    section = (ROOT / "templates" / f"readme-b2.{lang}.md").read_text().strip()
    section = (section.replace("{TABLES_27B}", tables("27b", lang))
               .replace("{TABLES_A3B}", tables("a3b", lang))
               .replace("{TAKEAWAYS}", (ROOT / "templates" / f"readme-b2-takeaways.{lang}.md").read_text().strip()))
    path = ROOT / name
    text = path.read_text()
    if START in text:
        head, rest = text.split(START, 1)
        tail = rest.split(END, 1)[1]
        text = head + START + "\n" + section + "\n" + END + tail
    else:
        assert switch in text, (name, switch)
        text = text.replace(switch, f"{switch}\n\n{START}\n{section}\n{END}\n\n{earlier}", 1)
    path.write_text(text)
    print(name, "updated")
