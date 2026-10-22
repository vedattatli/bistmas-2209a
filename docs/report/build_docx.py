"""Render docs/report/sonuc_raporu.md into the official TÜBİTAK 2209-A final-report template (.docx).

    uv run --with python-docx python docs/report/build_docx.py [out.docx]

The template (docs/report/2209a_sonuc_raporu_formati.docx, published by TÜBİTAK) keeps its title box, the
general-information table and the signature table; the "Sonuç Raporu Formatı" heading list is replaced by the
report body. Supported Markdown: the header key/value table, ## / ### headings, paragraphs, "* " bullets,
pipe tables, **bold** and images ``![caption](path relative to docs/report)``.
"""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import docx
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "2209a_sonuc_raporu_formati.docx"
SOURCE = HERE / "sonuc_raporu.md"
FONT = "Arial"


def parse(md: str) -> tuple[dict[str, str], str, list[str]]:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    title = re.search(r"\*\*PROJE BAŞLIĞI:\*\* \*\*(.+?)\*\*", md).group(1)
    info = dict(re.findall(r"^\| ([^|]+?) \| ([^|]+?) \|$", md.split("## 1.")[0], flags=re.M))
    body = md[md.index("## 1.") :].split("\n---\n")[0]
    return info, title, body.splitlines()


def add_runs(par, text: str, size: float = 11, bold: bool = False) -> None:
    for k, chunk in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if not chunk:
            continue
        run = par.add_run(chunk.replace("`", ""))
        run.bold = bold or k % 2 == 1
        run.font.name = FONT
        run.font.size = Pt(size)


def build(out: Path) -> None:
    info, title, lines = parse(SOURCE.read_text(encoding="utf-8"))
    doc = docx.Document(str(TEMPLATE))

    # title box: the template's placeholder texts live in w:t elements of a text frame
    executor = info.get("Proje yürütücüsünün adı", "").split(" (")[0]
    texts = list(doc.element.body.iter(qn("w:t")))
    for prev, t in zip([None, *texts], texts):
        if t.text and t.text.strip() == "ADI:" and prev is not None and (prev.text or "").endswith("YÜRÜTÜCÜSÜNÜN"):
            t.text = f" ADI: {executor}"  # the template splits this label over two runs
    for t in doc.element.body.iter(qn("w:t")):
        if t.text and "(koyu harflerle yazılacak)" in t.text:
            t.text = t.text.replace("(koyu harflerle yazılacak)", title)
        elif t.text and t.text.strip() == "PROJE YÜRÜTÜCÜSÜNÜN ADI:":
            t.text = f"PROJE YÜRÜTÜCÜSÜNÜN ADI: {info.get('Proje yürütücüsünün adı', '').split(' (')[0]}"
        elif t.text and t.text.strip() == "DANIŞMANININ ADI:":
            t.text = f" DANIŞMANININ ADI: {info.get('Danışmanın adı', '')}"

    # general-information table
    keys = {
        "PROJENİN KONUSU": "Projenin konusu",
        "PROJE YÜRÜTÜCÜSÜNÜN ADI": "Proje yürütücüsünün adı",
        "DANIŞMANIN ADI": "Danışmanın adı",
        "PROJE  BAŞLANGIÇ VE BİTİŞ TARİHLERİ": "Proje başlangıç ve bitiş tarihleri",
    }
    for row in doc.tables[0].rows:
        label = row.cells[0].text.strip()
        if label in keys:
            cell = row.cells[1]
            add_runs(cell.paragraphs[0], info.get(keys[label], ""), size=10)

    # replace the "Sonuç Raporu Formatı" heading list with the body
    paras = doc.paragraphs
    start = next(i for i, p in enumerate(paras) if p.text.strip().startswith("Sonuç Raporu Formatı"))
    end = next(i for i, p in enumerate(paras) if p.text.strip().startswith("Proje ile ilgili harcama"))
    anchor = paras[start]._p
    for p in paras[start + 1 : end + 1]:
        p._p.getparent().remove(p._p)
    blank = copy.deepcopy(anchor)

    def new_par():
        p = copy.deepcopy(blank)
        for child in list(p):
            if child.tag != qn("w:pPr"):
                p.remove(child)
        cursor[0].addnext(p)
        cursor[0] = p
        return docx.text.paragraph.Paragraph(p, doc._body)

    cursor = [anchor]
    anchor_par = docx.text.paragraph.Paragraph(anchor, doc._body)
    for r in list(anchor_par.runs):
        r._r.getparent().remove(r._r)

    buf: list[str] = []
    table: list[list[str]] = []

    def flush_par():
        if buf:
            add_runs(new_par(), " ".join(s.strip() for s in buf))
            buf.clear()

    def flush_table():
        if not table:
            return
        rows = [r for r in table if not re.fullmatch(r"[-| ]+", "|".join(r))]
        t = doc.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = doc.tables[0].style
        for i, r in enumerate(rows):
            for j, val in enumerate(r):
                add_runs(t.cell(i, j).paragraphs[0], val, size=9, bold=i == 0)
        cursor[0].addnext(t._tbl)
        cursor[0] = t._tbl
        table.clear()

    for line in lines:
        if line.startswith("|"):
            flush_par()
            table.append([c.strip() for c in line.strip().strip("|").split("|")])
            continue
        flush_table()
        if line.startswith("## "):
            flush_par()
            add_runs(new_par(), line[3:], size=12, bold=True)
        elif line.startswith("### "):
            flush_par()
            add_runs(new_par(), line[4:], size=11, bold=True)
        elif m := re.match(r"!\[(.*)\]\((.+)\)", line.strip()):
            flush_par()
            img = (HERE / m.group(2)).resolve()
            new_par().add_run().add_picture(str(img), width=Cm(16))
            add_runs(new_par(), m.group(1), size=9)
        elif line.startswith("* "):
            flush_par()
            buf.append("• " + line[2:])
        elif line.startswith("  ") and buf:
            buf.append(line)
        elif not line.strip():
            flush_par()
        else:
            buf.append(line)
    flush_par()
    flush_table()
    doc.save(str(out))
    print("written", out)


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "sonuc_raporu.docx")
