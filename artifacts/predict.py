"""Generate one XLSX with a frozen V4 package; change clarity cells only."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np
import openpyxl
from openpyxl.utils import get_column_letter

from .inference import Inference,CLASSES,digest
from .xlsx_preserve import write_labels,audit_output


def read_input(path,sheet,expected_rows):
    workbook=openpyxl.load_workbook(path,data_only=False,read_only=True)
    try:
        ws=workbook[sheet]
        header=[c.value for c in next(ws.iter_rows(min_row=1,max_row=1))]
        if header.count("resp_text")!=1 or header.count("clarity")!=1:raise ValueError("Expected unique resp_text/clarity headers")
        tc,lc=header.index("resp_text"),header.index("clarity")
        records=list(ws.iter_rows(min_row=2,values_only=True))
        if len(records)!=expected_rows:raise ValueError(f"Expected {expected_rows} data rows; got {len(records)}")
        if any(row[lc] not in (None,"") for row in records):raise ValueError("Input clarity must be empty")
        return [str(row[tc]) for row in records],get_column_letter(lc+1),list(range(2,len(records)+2))
    finally:workbook.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--sheet",default="test2")
    parser.add_argument("--expected-rows",type=int,default=897)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--package",type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    if args.output.exists() or args.input.resolve()==args.output.resolve():raise ValueError("Output must be new and distinct")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();input_sha=digest(args.input);manifest_sha=digest(args.package/"manifest.json")
    model=Inference(args.package,args.device)
    texts,column,excel_rows=read_input(args.input,args.sheet,args.expected_rows)
    p,components=model.predict_proba(texts,args.batch_size,return_components=True)
    predicted=[CLASSES[i] for i in p.argmax(axis=1)]
    mapping={f"{column}{row}":label for row,label in zip(excel_rows,predicted,strict=True)}
    pending=args.output.with_name(args.output.stem+".pending.xlsx")
    write_labels(args.input,pending,args.sheet,mapping)
    audit=audit_output(args.input,pending,args.sheet,mapping)
    if digest(args.input)!=input_sha:raise ValueError("Input file changed during inference")
    pending.replace(args.output)
    aux=args.output.with_suffix(".probabilities.npz")
    if aux.exists():raise FileExistsError(aux)
    np.savez_compressed(aux,source_row_ids=np.arange(len(texts)),excel_rows=np.array(excel_rows),probabilities=p,
                        **{f"{c}_probabilities":v for c,v in components.items()},class_order=np.array(CLASSES),
                        input_sha256=np.array(input_sha),package_manifest_sha256=np.array(manifest_sha))
    metadata={"status":"complete","rows":len(texts),"sheet":args.sheet,"class_order":list(CLASSES),
              "input_sha256":input_sha,"output_sha256":digest(args.output),"package_manifest_sha256":manifest_sha,
              "probabilities_sha256":digest(aux),"class_counts":dict(Counter(predicted)),"accuracy":None,
              "class_count_warning":"Predicted class counts are not accuracy; no candidate selection on this input",
              "cell_and_zip_audit":audit,"wall_seconds":time.perf_counter()-started,
              "completed_utc":datetime.now(timezone.utc).isoformat()}
    meta_path=args.output.with_suffix(".metadata.json")
    with meta_path.open("x",encoding="utf-8") as stream:json.dump(metadata,stream,ensure_ascii=False,indent=2);stream.write("\n")
    print(json.dumps(metadata,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
