"""Wrap the artifact fragment into a standalone page for GitHub Pages.

site/index.html is authored for the Artifact publisher, which supplies the
<!doctype>, <html>, <head> and <body> wrapper at publish time. GitHub Pages
serves files verbatim, so the same content needs a real document around it.
Keeping one source and generating the other avoids the two drifting apart.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

SRC = Path("site")
DST = Path("docs")
DESCRIPTION = ("A ballet variation extracted from video, fitted to a cosine "
               "series, and reconstructed live in the browser.")


def main() -> None:
    DST.mkdir(exist_ok=True)
    fragment = (SRC / "index.html").read_text()

    title_match = re.search(r"<title>(.*?)</title>", fragment, re.S)
    title = title_match.group(1).strip() if title_match else "Dancing"

    # everything through </style> belongs in <head>; the rest is body content
    split = fragment.index("</style>") + len("</style>")
    head_part = fragment[:split].replace(title_match.group(0), "", 1)
    body_part = fragment[split:]

    favicon = ("data:image/svg+xml,"
               "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
               "<text y='.9em' font-size='90'>%F0%9F%A9%B0</text></svg>")

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{DESCRIPTION}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{DESCRIPTION}">
<meta property="og:type" content="article">
<link rel="icon" href="{favicon}">
{head_part}
</head>
<body>
{body_part}
</body>
</html>
"""
    (DST / "index.html").write_text(page)
    for asset in ("data.js", "data3d.js"):
        shutil.copy2(SRC / asset, DST / asset)
    (DST / ".nojekyll").write_text("")

    size = sum(f.stat().st_size for f in DST.iterdir()) / 1024
    print(f"built {DST}/ — {len(list(DST.iterdir()))} files, {size:.0f} KB")


if __name__ == "__main__":
    main()
