#!/usr/bin/env python3
"""
Build the bioRxiv submission PDFs from docs/paper.md.

This encodes the inject-and-split pipeline that produces the two submission
artifacts, so they are regenerable rather than the output of an ad-hoc,
uncommitted set of manual steps:

  1. INJECT   - embed Figs 1-4 immediately above their captions in the main text
                (docs/paper.md carries captions only; the images live under
                results/submission_figures/, which is gitignored).
  2. SPLIT    - move Fig. 5 and its caption out of the main text into a separate
                supplementary document (bioRxiv wants supplementary material as
                its own file).
  3. RENDER   - pandoc + xelatex, US Letter, with the Unicode used in the prose
                (chi, rho, arrows, superscripts, R-hat) mapped to proper math so
                no glyph is silently dropped.

Usage:
    python scripts/build_paper.py
    python scripts/build_paper.py --figures-dir results/submission_figures \
                                  --outdir results/submission

Requires: pandoc and xelatex (TeX Live / MacTeX) on PATH.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Figure N -> basename under --figures-dir. Vector (.pdf) is preferred over .png.
FIGURES = {
    1: "fig1_internal_km",
    2: "fig2_external_km",
    3: "fig3_external_bor",
    4: "fig4_landmark_reversal",
    5: "fig5_pvalue_distributions",
}
SUPPLEMENTARY_FIGURES = {5}

SUPPLEMENTARY_TITLE = (
    "# Supplementary Figure --- Transportable survival, non-transportable response\n"
)

# Unicode that appears in the prose but is not safe to hand raw to LaTeX.
LATEX_HEADER = r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{microtype}
\usepackage{setspace}
\onehalfspacing
\setlength{\emergencystretch}{3em}
\usepackage{array}
\usepackage{newunicodechar}
\newunicodechar{χ}{\ensuremath{\chi}}
\newunicodechar{ρ}{\ensuremath{\rho}}
\newunicodechar{θ}{\ensuremath{\theta}}
\newunicodechar{σ}{\ensuremath{\sigma}}
\newunicodechar{μ}{\ensuremath{\mu}}
\newunicodechar{β}{\ensuremath{\beta}}
\newunicodechar{γ}{\ensuremath{\gamma}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{↓}{\ensuremath{\downarrow}}
\newunicodechar{↑}{\ensuremath{\uparrow}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{§}{\S}
\newunicodechar{⁻}{\ensuremath{{}^{-}}}
\newunicodechar{⁰}{\ensuremath{{}^{0}}}
\newunicodechar{¹}{\ensuremath{{}^{1}}}
\newunicodechar{²}{\ensuremath{{}^{2}}}
\newunicodechar{³}{\ensuremath{{}^{3}}}
\newunicodechar{⁵}{\ensuremath{{}^{5}}}
\newunicodechar{⁶}{\ensuremath{{}^{6}}}
\newunicodechar{⁸}{\ensuremath{{}^{8}}}
\newunicodechar{⁹}{\ensuremath{{}^{9}}}
\newunicodechar{–}{--}
\newunicodechar{—}{---}
"""


def resolve_figure(figures_dir: Path, basename: str) -> Path:
    """Prefer the vector PDF; fall back to the 300-DPI PNG."""
    for ext in (".pdf", ".png"):
        candidate = figures_dir / f"{basename}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No figure found for '{basename}' in {figures_dir} (looked for .pdf, .png). "
        "Run scripts/export_submission_figures.py first."
    )


def caption_pattern(n: int) -> re.Pattern:
    """Match the '**Figure N ...**' caption block through to the next blank-line break."""
    return re.compile(
        rf"^(\*\*Figure {n}\b.*?)(?=\n\n(?:\*\*Figure |#|---|\Z))",
        re.DOTALL | re.MULTILINE,
    )


def normalise_math(text: str) -> str:
    """Convert the one combining accent (R-hat) to explicit math.

    A raw combining circumflex over 'R' is fragile in xelatex and renders
    inconsistently; \\hat{R} is unambiguous.
    """
    return text.replace("R\u0302", r"$\hat{R}$")


def split_and_inject(md: str, figures_dir: Path) -> tuple[str, str]:
    """Return (main_markdown, supplementary_markdown)."""
    supp_blocks: list[str] = []

    for n, basename in FIGURES.items():
        pat = caption_pattern(n)
        match = pat.search(md)
        if not match:
            raise ValueError(
                f"Could not find the caption anchor for Figure {n} "
                f"(expected a line starting '**Figure {n}'). "
                "Has docs/paper.md been restructured?"
            )

        caption = match.group(1).rstrip()
        img_path = resolve_figure(figures_dir, basename)
        embed = f"![]({img_path.as_posix()})\\\n"

        if n in SUPPLEMENTARY_FIGURES:
            # Pull the figure and its caption OUT of the main document.
            supp_blocks.append(f"{embed}\n{caption}\n")
            md = pat.sub("", md, count=1)
        else:
            # Place the image directly above its caption.
            md = pat.sub(lambda m: f"{embed}\n{caption}\n", md, count=1)

    # Tidy any blank-line runs left behind by the removal.
    md = re.sub(r"\n{4,}", "\n\n\n", md)
    supplementary = SUPPLEMENTARY_TITLE + "\n" + "\n".join(supp_blocks)
    return md, supplementary


def render(md: str, out_pdf: Path, header: Path) -> None:
    """Render markdown -> PDF via pandoc + xelatex, patching the template if the
    (minimal-TeX-only) lmodern dependency is missing."""
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
        fh.write(md)
        src = Path(fh.name)

    base_cmd = [
        "pandoc", str(src), "-o", str(out_pdf),
        "--pdf-engine=xelatex",
        "-H", str(header),
        "-V", "fontsize=11pt",
        "-V", "mainfont=TeX Gyre Termes",
        "-V", "monofont=TeX Gyre Cursor",
        "-V", "geometry:letterpaper",
        "-V", "geometry:margin=1in",
        "-V", "linkcolor=blue",
    ]

    proc = subprocess.run(base_cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        src.unlink(missing_ok=True)
        return

    # Minimal TeX installs (e.g. CI containers) lack lmodern.sty, which pandoc's
    # default template loads unconditionally. Full TeX Live / MacTeX ships it, so
    # this branch is a fallback, not the normal path.
    if "lmodern.sty" not in proc.stderr + proc.stdout:
        src.unlink(missing_ok=True)
        sys.exit(f"pandoc failed:\n{proc.stdout}\n{proc.stderr}")

    tmpl = Path(tempfile.mkdtemp()) / "default.tex"
    tmpl.write_text(
        subprocess.run(["pandoc", "-D", "latex"], capture_output=True, text=True)
        .stdout.replace(r"\usepackage{lmodern}", "% lmodern unavailable; using default CM")
    )
    proc = subprocess.run(base_cmd + ["--template", str(tmpl)], capture_output=True, text=True)
    src.unlink(missing_ok=True)
    if proc.returncode != 0:
        sys.exit(f"pandoc failed (with patched template):\n{proc.stdout}\n{proc.stderr}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", type=Path, default=Path("docs/paper.md"))
    ap.add_argument("--figures-dir", type=Path, default=Path("results/submission_figures"))
    ap.add_argument("--outdir", type=Path, default=Path("results/submission"))
    args = ap.parse_args()

    for tool in ("pandoc", "xelatex"):
        if shutil.which(tool) is None:
            sys.exit(f"'{tool}' not found on PATH. Install pandoc and a TeX distribution "
                     "(MacTeX on macOS, TeX Live on Linux).")

    if not args.paper.exists():
        sys.exit(f"Manuscript not found: {args.paper}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    md = normalise_math(args.paper.read_text())
    main_md, supp_md = split_and_inject(md, args.figures_dir)

    header = args.outdir / "_header.tex"
    header.write_text(LATEX_HEADER)

    main_pdf = args.outdir / "paper.pdf"
    supp_pdf = args.outdir / "paper_supplementary.pdf"
    render(main_md, main_pdf, header)
    render(supp_md, supp_pdf, header)
    header.unlink(missing_ok=True)

    print(f"  main manuscript  -> {main_pdf}  (Figs 1-4 embedded)")
    print(f"  supplementary    -> {supp_pdf}  (Fig. 5)")
    print("\nBoth are build artifacts under results/ and are gitignored by design;")
    print("this script is the committed recipe that regenerates them.")


if __name__ == "__main__":
    main()
