#!/usr/bin/env python3
"""Prepare a filtered, language-labelled corpus for the existing Purple AI V1.0."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path

EXTENSIONS = {
    ".py": "Python", ".pyw": "Python", ".js": "JavaScript",
    ".mjs": "JavaScript", ".cjs": "JavaScript", ".ts": "TypeScript",
    ".tsx": "TSX", ".jsx": "JSX", ".java": "Java", ".kt": "Kotlin",
    ".kts": "Kotlin", ".swift": "Swift", ".c": "C", ".h": "C/C++",
    ".cc": "C++", ".cpp": "C++", ".cxx": "C++", ".hpp": "C++",
    ".cs": "C#", ".go": "Go", ".rs": "Rust", ".rb": "Ruby",
    ".php": "PHP", ".lua": "Lua", ".dart": "Dart", ".scala": "Scala",
    ".r": "R", ".jl": "Julia", ".pl": "Perl", ".hs": "Haskell",
    ".ex": "Elixir", ".exs": "Elixir", ".erl": "Erlang",
    ".clj": "Clojure", ".fs": "F#", ".vb": "Visual Basic",
    ".ps1": "PowerShell", ".sh": "Shell", ".bash": "Bash",
    ".sql": "SQL", ".html": "HTML", ".htm": "HTML", ".css": "CSS",
    ".scss": "SCSS", ".sass": "Sass", ".vue": "Vue",
    ".svelte": "Svelte", ".xml": "XML", ".json": "JSON",
    ".yml": "YAML", ".yaml": "YAML", ".toml": "TOML",
    ".graphql": "GraphQL", ".gql": "GraphQL", ".sol": "Solidity",
    ".move": "Move", ".v": "Verilog", ".vhd": "VHDL",
    ".vhdl": "VHDL", ".m": "MATLAB/Octave", ".f": "Fortran",
    ".f90": "Fortran", ".cob": "COBOL", ".asm": "Assembly",
    ".s": "Assembly", ".md": "Markdown", ".txt": "Text/config",
    ".ini": "INI", ".cfg": "Config", ".conf": "Config",
    ".dockerfile": "Dockerfile",
}

SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", "vendor",
    "dist", "build", "target", ".venv", "venv", "env", ".idea",
    ".vscode", "coverage", ".next", ".nuxt", "site-packages",
}
SECRET_NAME = re.compile(
    r"(secret|credential|\.env($|\.)|id_rsa|id_ed25519|\.pem$|\.key$|"
    r"token|password|passwd|\.p12$|\.pfx$)",
    re.IGNORECASE,
)
SECRET_VALUE = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*['\"]?[^\s'\"]{8,}"
)
MAX_FILE_BYTES = 256_000
MIN_CHARS = 40


def language_for(path: Path) -> str | None:
    name = path.name.lower()
    if name == "dockerfile":
        return "Dockerfile"
    if name == "makefile":
        return "Makefile"
    return EXTENSIONS.get(path.suffix.lower())


def clean_source(text: str) -> str:
    text = text.replace("\x00", "")
    lines = []
    for line in text.splitlines():
        # Do not include lines that look like they contain credentials.
        if SECRET_VALUE.search(line):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True,
                        help="Folder with code; repeat this option for more folders.")
    parser.add_argument("--output", default="training/programming_data.txt")
    parser.add_argument("--max-files", type=int, default=100000)
    parser.add_argument("--max-file-bytes", type=int, default=MAX_FILE_BYTES)
    args = parser.parse_args()

    roots = [Path(p).expanduser().resolve() for p in args.source]
    for root in roots:
        if not root.is_dir():
            raise SystemExit(f"Source folder does not exist: {root}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    written = 0
    skipped = 0
    total_bytes = 0

    with output.open("w", encoding="utf-8") as out:
        for root in roots:
            for current, dirs, filenames in os.walk(root):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                for filename in sorted(filenames):
                    path = Path(current) / filename
                    language = language_for(path)
                    if not language or SECRET_NAME.search(filename):
                        skipped += 1
                        continue
                    try:
                        size = path.stat().st_size
                        if size <= 0 or size > args.max_file_bytes:
                            skipped += 1
                            continue
                        raw = path.read_text(encoding="utf-8", errors="strict")
                        source = clean_source(raw)
                        if len(source) < MIN_CHARS:
                            skipped += 1
                            continue
                        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
                        if digest in seen_hashes:
                            skipped += 1
                            continue
                        seen_hashes.add(digest)

                        relative = path.name
                        block = (
                            "\n\n### JĘZYK: " + language + "\n"
                            "### PLIK: " + relative + "\n"
                            "### ZADANIE: Przeanalizuj i kontynuuj kod zgodnie z jego składnią.\n"
                            "### KOD:\n" + source + "\n"
                            "### KONIEC KODU\n"
                        )
                        out.write(block)
                        written += 1
                        total_bytes += len(block.encode("utf-8"))
                        if written % 100 == 0:
                            print(f"Przygotowano plików: {written} | pominięto: {skipped}", flush=True)
                        if written >= args.max_files:
                            break
                    except (OSError, UnicodeError):
                        skipped += 1
                if written >= args.max_files:
                    break

    print("\nGotowe.")
    print(f"Pliki użyte: {written}")
    print(f"Pliki pominięte: {skipped}")
    print(f"Rozmiar korpusu: {total_bytes:,} bajtów")
    print(f"Zapisano: {output.resolve()}")


if __name__ == "__main__":
    main()
