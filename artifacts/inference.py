"""Portable, local-only inference for the frozen V4 triple-stacking package."""
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from transformers import AutoModelForSequenceClassification,AutoTokenizer

CLASSES=("c1","c234","c5")


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def probabilities(p,n):
    p=np.asarray(p,dtype=np.float64)
    if p.shape!=(n,3) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(axis=1),1,atol=1e-6,rtol=0):
        raise ValueError("Invalid three-class probabilities")
    return p


class Inference:
    def __init__(self,directory=None,device="cpu"):
        self.directory=Path(directory) if directory else Path(__file__).resolve().parent
        manifest=json.loads((self.directory/"manifest.json").read_text(encoding="utf-8"))
        for name,expected in manifest["files"].items():
            path=(self.directory/name).resolve()
            if not path.is_relative_to(self.directory.resolve()) or digest(path)!=expected:raise ValueError(f"Package integrity mismatch: {name}")
        config=json.loads((self.directory/"inference_config.json").read_text(encoding="utf-8"))
        if config["class_order"]!=list(CLASSES) or config["component_order"]!=["bert","word","char"] or config["family"]!="stack_three":
            raise ValueError("Frozen inference contract differs")
        torch.set_num_threads(4)
        if device.startswith("cuda"):
            if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; use --device cpu")
            torch.backends.cuda.matmul.fp32_precision="ieee"
            torch.backends.cudnn.conv.fp32_precision="ieee"
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
            torch.use_deterministic_algorithms(True)
        self.device=torch.device(device)
        self.tokenizer=AutoTokenizer.from_pretrained(self.directory/"bert",local_files_only=True)
        self.bert=AutoModelForSequenceClassification.from_pretrained(self.directory/"bert",local_files_only=True).to(self.device).eval()
        if [self.bert.config.id2label[i] for i in range(3)]!=list(CLASSES):raise ValueError("BERT classes differ")
        self.word=joblib.load(self.directory/"word.joblib")
        self.char=joblib.load(self.directory/"char.joblib")
        self.stacker=joblib.load(self.directory/"stacker.joblib")
        for model in (self.word,self.char,self.stacker):
            if list(model.classes_)!=list(CLASSES):raise ValueError("Lexical/stacker classes differ")
        if self.stacker.n_features_in_!=9:raise ValueError("Expected nine raw stacker features")

    def predict_proba(self,texts,batch_size=4,return_components=False):
        if batch_size<1:raise ValueError("batch_size must be positive")
        texts=[t if isinstance(t,str) else str(t) for t in texts]
        n=len(texts)
        if not n:
            empty=np.empty((0,3));return (empty,{c:empty.copy() for c in ("bert","word","char")}) if return_components else empty
        out=[]
        with torch.inference_mode():
            for start in range(0,n,batch_size):
                tokens=self.tokenizer(texts[start:start+batch_size],truncation=True,max_length=512,padding=True,return_tensors="pt")
                logits=self.bert(**{k:v.to(self.device) for k,v in tokens.items()}).logits
                out.append(torch.softmax(logits,dim=-1).cpu().numpy())
        components={"bert":probabilities(np.concatenate(out),n)}
        with threadpool_limits(limits=4):
            components["word"]=probabilities(self.word.predict_proba(texts),n)
            components["char"]=probabilities(self.char.predict_proba(texts),n)
            features=np.concatenate([components[c] for c in ("bert","word","char")],axis=1)
            final=probabilities(self.stacker.predict_proba(features),n)
        return (final,components) if return_components else final

    def predict(self,texts,batch_size=4):
        return [CLASSES[i] for i in self.predict_proba(texts,batch_size).argmax(axis=1)]
