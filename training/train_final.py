"""Approved V4 final triple stack, original-base components in A_fit, stacker in B only."""
from datetime import datetime,timezone
import gc
from pathlib import Path
import shutil
import sys
import time

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parent/"scripts"))
import bert_engine
import lexical
from ensembles import stacker
from common import CLASSES,V4,digest,read_train,write_json,deduplicate_fit
from runtime import FROZEN,atomic_json,environment,execution_lock,implementation,json_read,object_sha,save_probabilities,load_probabilities,verify_files


def main():
    sources=json_read(V4/"final/training_sources_manifest.json")
    verify_files(V4,sources["files"])
    if environment()!=sources["environment"]:raise ValueError("Final environment changed")
    final=V4/"final/training"
    final.mkdir(parents=True,exist_ok=True)
    if (final/"complete.json").exists():
        verify_files(final,json_read(final/"complete.json")["files"])
        print("Final components already complete and verified",flush=True);return
    with execution_lock(V4/"final/training.lock"):
        started=time.perf_counter()
        split=json_read(V4/"final/final_split.json")
        rows=read_train()
        a,b,af=split["A"],split["B"],split["A_fit"]
        if set(a)&set(b) or set(a)|set(b)!=set(range(len(rows))):raise ValueError("Final row split invalid")
        if {rows[i]["group"] for i in a}&{rows[i]["group"] for i in b}:raise ValueError("Final group leakage")
        if [r["row_id"] for r in deduplicate_fit([rows[i] for i in a])]!=af:raise ValueError("Final A_fit deduplication invalid")
        for ids in (af,b):
            if {rows[i]["label"] for i in ids}!=set(CLASSES):raise ValueError("Final missing class")
        meta={"partition":"final_B","class_order":list(CLASSES),"fit_ids_sha256":object_sha(af),
              "split_sha256":digest(V4/"final/final_split.json"),"protocol_sha256":digest(FROZEN/"protocol.json"),
              "sources_manifest_sha256":digest(V4/"final/training_sources_manifest.json"),"environment":environment()}
        atomic_json(final/"progress.json",{"stage":"bert","updates_required":3557})
        model,tokenizer,result=bert_engine.train(rows,af,final/"bert")
        probabilities={}
        path=final/"bert_B.npz"
        component_meta={**meta,"component":"bert"}
        if path.exists():probabilities["bert"]=load_probabilities(path,b,component_meta)
        else:
            probabilities["bert"]=bert_engine.predict(model,tokenizer,[rows[i]["text"] for i in b])
            save_probabilities(path,b,probabilities["bert"],component_meta)
        del model,tokenizer;gc.collect();torch.cuda.empty_cache()
        for component in ("word","char"):
            atomic_json(final/"progress.json",{"stage":component})
            model_path=final/f"{component}.joblib"
            if model_path.exists():
                linear=joblib.load(model_path);messages=[]
            else:
                linear,messages=lexical.fit(component,rows,af)
                tmp=model_path.with_suffix(".tmp");joblib.dump(linear,tmp);tmp.replace(model_path)
            audit=lexical.audit_fit(linear,component,rows,af)
            audit_path=final/f"{component}_audit.json"
            if not audit_path.exists():write_json(audit_path,{**audit,"warnings":messages})
            path=final/f"{component}_B.npz";component_meta={**meta,"component":component}
            if path.exists():probabilities[component]=load_probabilities(path,b,component_meta)
            else:
                probabilities[component]=lexical.predict(linear,[rows[i]["text"] for i in b])
                save_probabilities(path,b,probabilities[component],component_meta)
        atomic_json(final/"progress.json",{"stage":"stacker_B_only"})
        features=np.concatenate([probabilities[c] for c in ("bert","word","char")],axis=1)
        y=[rows[i]["label"] for i in b]
        with threadpool_limits(limits=4):
            combined=stacker().fit(features,y)
            fitted_probabilities=combined.predict_proba(features)
        if list(combined.classes_)!=list(CLASSES):raise ValueError("Final stacker class order mismatch")
        joblib.dump(combined,final/"stacker.joblib")
        save_path=final/"stacker_B_fit_probabilities.npz"
        if not save_path.exists():save_probabilities(save_path,b,fitted_probabilities,{**meta,"component":"stack_three","fit_partition":"B","scope":"stacker training probabilities; not generalization"})
        write_json(final/"training_summary.json",{"family":"stack_three","A":len(a),"A_fit":len(af),"B":len(b),
            "bert_updates":3557,"word_fits":1,"char_fits":1,"stacker_fits":1,"stacker_C":1,
            "class_order":list(CLASSES),"stacker_features":"bert,word,char; three raw probabilities each",
            "no_A_plus_B_refit":True,"original_base_initialization":True,
            "full_optimizer_state_retained":True,"wall_seconds":time.perf_counter()-started,
            "completed_utc":datetime.now(timezone.utc).isoformat(),"official_test_access":False})
        atomic_json(final/"progress.json",{"stage":"complete"})
        write_json(final/"complete.json",{"files":{str(p.relative_to(final)):digest(p) for p in sorted(final.rglob("*")) if p.is_file() and p.name!="complete.json"},
            "source_manifest_sha256":digest(V4/"final/training_sources_manifest.json"),"bert_updates":3557,
            "completed_utc":datetime.now(timezone.utc).isoformat()})
        print("FINAL TRAINING COMPLETE: additional BERT + word + char + B-only stacker; no official test access",flush=True)


if __name__=="__main__":main()
