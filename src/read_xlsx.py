"""Minimal .xlsx reader (stdlib only): prints every sheet as rows of cell values."""
import sys, zipfile, re, xml.etree.ElementTree as ET
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

def col_idx(ref):
    """Zero-based column index from an Excel cell reference such as 'C12'."""
    letters = re.match(r"[A-Z]+", ref).group(0); n = 0
    for ch in letters: n = n * 26 + ord(ch) - 64
    return n - 1

def read(path):
    """Read the first worksheet of an .xlsx file into a list of rows (standard library only)."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    target = {r.get("Id"): r.get("Target") for r in rels}
    out = {}
    for s in wb.find("m:sheets", NS):
        t = target[s.get(RNS)]; t = t.lstrip("/"); t = t if t.startswith("xl/") else "xl/" + t
        rows = []
        for r in ET.fromstring(z.read(t)).iter("{%s}row" % NS["m"]):
            vals = {}
            for c in r.findall("m:c", NS):
                v = c.find("m:v", NS); typ = c.get("t")
                if typ == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter("{%s}t" % NS["m"]))
                elif v is None: continue
                elif typ == "s": val = shared[int(v.text)]
                else: val = v.text
                vals[col_idx(c.get("r"))] = val
            if vals: rows.append([vals.get(i) for i in range(max(vals) + 1)])
        out[s.get("name")] = rows
    return out

if __name__ == "__main__":
    for name, rows in read(sys.argv[1]).items():
        print("== sheet", name, len(rows), "rows")
        for i, r in enumerate(rows):
            if i < 12 or i >= len(rows) - 4: print(i, r)
