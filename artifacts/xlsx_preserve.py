"""Restrict XLSX changes to clarity cells inside the single target worksheet XML."""
import copy
from pathlib import Path,PurePosixPath
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

MAIN="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG="http://schemas.openxmlformats.org/package/2006/relationships"
NS={"m":MAIN}
CLASSES={"c1","c234","c5"}


def sheet_member(archive,sheet):
    book=ET.fromstring(archive.read("xl/workbook.xml"))
    matches=[s for s in book.findall("m:sheets/m:sheet",NS) if s.attrib["name"]==sheet]
    if len(matches)!=1:raise ValueError("Target worksheet not found uniquely")
    rel_id=matches[0].attrib[f"{{{REL}}}id"]
    relationships=ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    target=next(x.attrib["Target"] for x in relationships if x.attrib["Id"]==rel_id)
    member=target.lstrip("/") if target.startswith("/") else str(PurePosixPath("xl")/target)
    if ".." in PurePosixPath(member).parts or member not in archive.namelist():raise ValueError("Unsafe worksheet member")
    return member


def column_number(reference):
    result=0
    for char in re.match(r"[A-Z]+",reference).group():result=result*26+ord(char)-64
    return result


def label_cell(original,reference,label,prefix=""):
    if label not in CLASSES:raise ValueError("Invalid class label")
    if original:
        opening=re.match(r"<((?:[\w.-]+:)?c)\b([^>]*?)(/?)>",original)
        if not opening:raise ValueError("Unsupported cell XML")
        qname,attrs=opening.group(1),opening.group(2)
        attrs=re.sub(r"\s+t\s*=\s*(?:\"[^\"]*\"|'[^']*')","",attrs)
        prefix=qname[:-1]
    else:qname=prefix+"c";attrs=f' r="{reference}"'
    return f'<{qname}{attrs} t="inlineStr"><{prefix}is><{prefix}t>{label}</{prefix}t></{prefix}is></{qname}>'


def replace_sheet(xml_bytes,labels):
    text=xml_bytes.decode("utf-8")
    for reference,label in labels.items():
        row_number=int(re.search(r"\d+$",reference).group())
        cell_pattern=re.compile(r"<(?:[\w.-]+:)?c\b(?=[^>]*\br\s*=\s*['\"]"+re.escape(reference)+r"['\"])[^>]*?(?:/>|>.*?</(?:[\w.-]+:)?c\s*>)",re.S)
        found=list(cell_pattern.finditer(text))
        if len(found)>1:raise ValueError("Duplicate cell reference")
        if found:
            match=found[0];text=text[:match.start()]+label_cell(match.group(),reference,label)+text[match.end():]
            continue
        row_pattern=re.compile(r"(<((?:[\w.-]+:)?row)\b(?=[^>]*\br\s*=\s*['\"]"+str(row_number)+r"['\"])[^>]*?)(?:/>|>(.*?)</\2\s*>)",re.S)
        row=row_pattern.search(text)
        if row:
            prefix=row.group(2)[:-3];body=row.group(3) or ""
            insert=len(body)
            for cell in re.finditer(r"<(?:[\w.-]+:)?c\b[^>]*\br=['\"]([A-Z]+\d+)['\"][^>]*?(?:/>|>.*?</(?:[\w.-]+:)?c\s*>)",body,re.S):
                if column_number(cell.group(1))>column_number(reference):insert=cell.start();break
            body=body[:insert]+label_cell(None,reference,label,prefix)+body[insert:]
            replacement=row.group(1)+">"+body+"</"+row.group(2)+">"
            text=text[:row.start()]+replacement+text[row.end():]
        else:
            sheet_data=re.search(r"<((?:[\w.-]+:)?sheetData)\b[^>]*>(.*?)</\1\s*>",text,re.S)
            if not sheet_data:raise ValueError("Unsupported absent sheetData")
            prefix=sheet_data.group(1)[:-9];body=sheet_data.group(2);insert=len(body)
            for match in re.finditer(r"<(?:[\w.-]+:)?row\b[^>]*\br=['\"](\d+)['\"]",body):
                if int(match.group(1))>row_number:insert=match.start();break
            row_xml=f'<{prefix}row r="{row_number}">'+label_cell(None,reference,label,prefix)+f'</{prefix}row>'
            replacement=body[:insert]+row_xml+body[insert:]
            text=text[:sheet_data.start(2)]+replacement+text[sheet_data.end(2):]
    return text.encode("utf-8")


def write_labels(input_path,output_path,sheet,labels):
    input_path,output_path=Path(input_path),Path(output_path)
    if input_path.resolve()==output_path.resolve() or output_path.exists():raise ValueError("Output must be new and distinct")
    with ZipFile(input_path) as source:
        if len(source.namelist())!=len(set(source.namelist())):raise ValueError("Duplicate XLSX members")
        target=sheet_member(source,sheet)
        changed=replace_sheet(source.read(target),labels)
        with ZipFile(output_path,"w") as destination:
            destination.comment=source.comment
            for info in source.infolist():destination.writestr(info,changed if info.filename==target else source.read(info.filename))
    return target


def audit_output(input_path,output_path,sheet,labels):
    """Independent XML/cell audit; does not call the writer or its replacement function."""
    with ZipFile(input_path) as before,ZipFile(output_path) as after:
        if before.namelist()!=after.namelist() or before.comment!=after.comment:raise ValueError("ZIP structure/comment changed")
        target=sheet_member(before,sheet)
        changed=[name for name in before.namelist() if before.read(name)!=after.read(name)]
        if changed!=[target]:raise ValueError(f"Unexpected changed XLSX members: {changed}")
        x,y=ET.fromstring(before.read(target)),ET.fromstring(after.read(target))
        original_rows={r.attrib["r"] for r in x.findall("m:sheetData/m:row",NS)}
        for tree,is_after in ((x,False),(y,True)):
            seen=set()
            for row in list(tree.find("m:sheetData",NS)):
                for cell in list(row):
                    ref=cell.attrib.get("r")
                    if ref in labels:
                        if is_after:
                            if cell.attrib.get("t")!="inlineStr" or "".join(cell.itertext())!=labels[ref]:raise ValueError("Written class differs")
                            seen.add(ref)
                        row.remove(cell)
                if is_after and row.attrib.get("r") not in original_rows and len(row)==0:tree.find("m:sheetData",NS).remove(row)
            if is_after and seen!=set(labels):raise ValueError("Missing written label cells")
        if ET.tostring(x)!=ET.tostring(y):raise ValueError("Worksheet changed outside label cells")
    import openpyxl
    wb1=openpyxl.load_workbook(input_path,data_only=False);wb2=openpyxl.load_workbook(output_path,data_only=False)
    checked=0
    try:
        if wb1.sheetnames!=wb2.sheetnames:raise ValueError("Sheet names/order changed")
        for name in wb1.sheetnames:
            a,b=wb1[name],wb2[name]
            if (a.max_row,a.max_column)!=(b.max_row,b.max_column):raise ValueError("Sheet dimensions changed")
            for row in a:
                for original in row:
                    modified=b[original.coordinate]
                    if original.style_id!=modified.style_id:raise ValueError("Cell style changed")
                    expected=labels.get(original.coordinate,original.value) if name==sheet else original.value
                    if modified.value!=expected or (not(name==sheet and original.coordinate in labels) and modified.data_type!=original.data_type):raise ValueError("Cell value/type changed unexpectedly")
                    checked+=1
    finally:wb1.close();wb2.close()
    return {"verified":True,"only_changed_member":target,"label_cells":len(labels),"all_cells_checked":checked,"styles_preserved":True,"other_zip_members_byte_identical":True}
