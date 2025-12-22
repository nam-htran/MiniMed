import os
import re
from pathlib import Path

DUMP_FILE = "_utils/dump.txt"

def main():
    with open(DUMP_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Tách theo pattern: ===== .\path\to\file =====
    pattern = re.compile(r"===== \.\\(.+?) =====\n", re.MULTILINE)
    matches = list(pattern.finditer(content))

    for i, match in enumerate(matches):
        rel_path = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        file_content = content[start:end].lstrip("\n")

        out_path = Path(rel_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(file_content)

        print(f"✅ Written: {out_path}")

if __name__ == "__main__":
    main()
