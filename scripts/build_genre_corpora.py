#!/usr/bin/env python3
"""Rebuild corpora/*.txt for the genre bench (scenarios/g-*.json). The texts are third-party and not committed.

  python scripts/build_genre_corpora.py

- code-cpython-stdlib.txt: this Python's standard library sources (Lib/, test packages left out), PSF License.
- prose-en-trollope.txt: Anthony Trollope, "The Way We Live Now" (Project Gutenberg #5231), public domain;
  the Gutenberg header and footer are cut.
- prose-ja-daibosatsu.txt: 中里介山『大菩薩峠』01-08 (青空文庫, 著作権なし); the Aozora header, footer, ruby and
  annotations are removed.
- agent-swe-openhands.txt: the six longest of the first 200 trajectories of nebius/SWE-rebench-openhands-trajectories
  (CC-BY-4.0), rendered as a transcript (role, content, tool calls).
"""
import glob
import json
import re
import sysconfig
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "corpora"
RAW = ROOT / "raw"
AOZORA = ["1726_ruby_5424", "3226_ruby_1818", "3227_ruby_1846", "3228_ruby_1849", "3332_ruby_53", "3333_ruby_112",
          "3334_ruby_115", "3335_ruby_1858"]


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def code() -> str:
    lib = Path(sysconfig.get_paths()["stdlib"])
    parts = []
    for p in sorted(lib.rglob("*.py")):
        rel = p.relative_to(lib).as_posix()
        if any(seg in ("test", "tests", "idle_test", "site-packages") for seg in rel.split("/")) or \
                rel.startswith(("idlelib/", "turtledemo/", "lib2to3/")):
            continue
        try:
            parts.append(f"# ===== file: Lib/{rel} =====\n" + p.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            pass
    return "\n\n".join(parts)


def prose_en() -> str:
    t = fetch("https://www.gutenberg.org/cache/epub/5231/pg5231.txt", RAW / "trollope-way-we-live-now.txt").read_text(encoding="utf-8")
    a = t.index("\n", t.index("*** START OF THE PROJECT")) + 1
    return t[a:t.index("*** END OF THE PROJECT")].strip()


def prose_ja() -> str:
    vols = []
    for name in AOZORA:
        z = fetch(f"https://www.aozora.gr.jp/cards/000283/files/{name}.zip", RAW / "aozora" / f"{name}.zip")
        with zipfile.ZipFile(z) as zf:
            member = next(n for n in zf.namelist() if n.endswith(".txt"))
            s = zf.read(member).decode("cp932", errors="replace").replace("\r\n", "\n")
        title = s.split("\n", 1)[0].strip()
        seps = [m.start() for m in re.finditer(r"^-{20,}$", s, re.M)]
        if len(seps) >= 2:
            s = s[seps[1]:].split("\n", 1)[1]
        i = s.find("底本：")
        if i > 0:
            s = s[:i]
        s = re.sub(r"《[^》]*》", "", s).replace("｜", "")
        s = re.sub(r"［＃[^］]*］", "", s)
        vols.append(title + "\n\n" + s.strip())
    return "\n\n".join(vols)


def agent() -> str:
    rows = []
    for off in (0, 100):
        p = fetch("https://datasets-server.huggingface.co/rows?dataset=nebius/SWE-rebench-openhands-trajectories"
                  f"&config=default&split=train&offset={off}&length=100", RAW / "swe" / f"rows-{off}.json")
        rows += [r["row"] for r in json.load(p.open())["rows"]]

    def render(row):
        lines = [f"===== session {row['trajectory_id']} ({row['repo']}) ====="]
        for m in row["trajectory"]:
            body = m.get("content") or ""
            if m.get("tool_calls"):
                body += ("\n" if body else "") + "tool_calls: " + json.dumps(m["tool_calls"], ensure_ascii=False)
            lines.append(f"[{m.get('role')}]\n{body}")
        return "\n\n".join(lines)

    rows.sort(key=lambda r: -sum(len(m.get("content") or "") for m in r["trajectory"]))
    return "\n\n".join(render(r) for r in rows[:6])


if __name__ == "__main__":
    for name, fn in (("code-cpython-stdlib.txt", code), ("prose-en-trollope.txt", prose_en),
                     ("prose-ja-daibosatsu.txt", prose_ja), ("agent-swe-openhands.txt", agent)):
        (ROOT / name).write_text(fn(), encoding="utf-8")
        print(name, (ROOT / name).stat().st_size)
