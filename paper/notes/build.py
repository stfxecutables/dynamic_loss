from os import system


from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build.sh"
STYLE = str(ROOT / "mdstyle.html")
MDS = sorted(ROOT.rglob("notes.md"))
HTMLS = list(map(lambda p: Path(str(p).replace("notes.md", "notes.html")), MDS))

for md_path, html_path in zip(MDS, HTMLS):
    infile = str(md_path).replace(" ", "\ ")
    outfile = str(html_path).replace(" ", "\ ")
    system(f'pandoc --self-contained -f markdown -t html {infile} -s --metadata pagetitle="Notes" -H {STYLE} -o {outfile}')

