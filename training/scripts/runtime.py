"""V4 integrity, probability contracts and atomic progress records."""
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket

import numpy as np
from common import CLASSES, TRAIN_SHA256, V4, digest

FROZEN = V4 / "splits/frozen_v1"


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def implementation():
    return {p.name: digest(p) for p in sorted((V4 / "scripts").glob("*.py"))}


def environment():
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "tokenizers", "safetensors", "numpy", "scipy", "scikit-learn", "joblib", "openpyxl")}}


def verify_files(directory, manifest):
    for name, sha in manifest.items():
        path = (Path(directory) / name).resolve()
        if not path.is_relative_to(Path(directory).resolve()) or digest(path) != sha:
            raise ValueError(f"Artifact integrity mismatch: {name}")


def frozen_split(r, f):
    manifest = json_read(FROZEN / "manifest.json")
    if digest(FROZEN / "protocol.json") != manifest["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    name = f"r{r:02d}_f{f:02d}.json"
    record = next(x for x in manifest["splits"] if x["path"] == name)
    if digest(FROZEN / name) != record["sha256"]:
        raise ValueError("Frozen split changed")
    return json_read(FROZEN / name)


def provenance(split, fit_ids, component, partition):
    return {"schema": 1, "protocol_sha256": digest(FROZEN / "protocol.json"),
            "training_sha256": TRAIN_SHA256, "fit_ids_sha256": object_sha(fit_ids),
            "split_sha256": object_sha(split), "round": split["round"], "fold": split["fold"],
            "partition": partition, "component": component, "class_order": list(CLASSES),
            "implementation_sha256": object_sha(implementation()), "environment": environment()}


def validate_probabilities(p, count=None):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or (count is not None and len(p) != count):
        raise ValueError("Probability shape mismatch")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Invalid probability values/sums")
    return p


def save_probabilities(path, row_ids, p, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Duplicate probability row IDs")
    p = validate_probabilities(p, len(row_ids))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, row_ids=np.asarray(row_ids, dtype=np.int64), probabilities=p,
                            metadata=np.array(json.dumps(metadata, sort_keys=True)))
    os.replace(tmp, path)


def load_probabilities(path, row_ids, expected_metadata):
    with np.load(path, allow_pickle=False) as data:
        if data["row_ids"].tolist() != row_ids or json.loads(str(data["metadata"])) != expected_metadata:
            raise ValueError("Probability alignment/provenance mismatch")
        return validate_probabilities(data["probabilities"], len(row_ids))


@contextmanager
def execution_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse stale locks too: recovery must inspect the recorded PID before removal.
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "hostname": socket.gethostname()}, stream)
    try:
        yield
    finally:
        path.unlink()
