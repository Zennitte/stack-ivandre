"""Fixed V4 lexical recipes; vocabulary and IDF fit only on the supplied A_fit."""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from common import CLASSES
from runtime import FROZEN, json_read, object_sha, validate_probabilities


def build(component):
    cfg = json_read(FROZEN / "protocol.json")
    own = cfg[component]
    shared = {k: v for k, v in cfg["lexical_shared"].items() if k != "classifier"}
    shared["dtype"] = np.float64
    shared.update(analyzer=own["analyzer"], ngram_range=tuple(own["ngram_range"]))
    if component == "word":
        shared["token_pattern"] = own["token_pattern"]
    classifier = {k: v for k, v in cfg["lexical_shared"]["classifier"].items() if k != "multiclass"}
    classifier.update(C=own["C"], random_state=own["seed"], n_jobs=own["n_jobs"])
    return Pipeline([("vectorizer", TfidfVectorizer(**shared)), ("classifier", LogisticRegression(**classifier))])


def fit(component, rows, ids):
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated fit IDs")
    model = build(component)
    with threadpool_limits(limits=4), warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit([rows[i]["text"] for i in ids], [rows[i]["label"] for i in ids])
    if list(model.classes_) != list(CLASSES):
        raise ValueError("Lexical class order mismatch")
    # Store exact training identity; audit vocabulary/IDF independently by fresh reconstruction.
    model.v4_fit_ids_sha256 = object_sha(ids)
    return model, [str(w.message) for w in observed]


def predict(model, texts):
    order = [list(model.classes_).index(c) for c in CLASSES]
    with threadpool_limits(limits=4):
        return validate_probabilities(model.predict_proba(texts)[:, order], len(texts))


def audit_fit(model, component, rows, ids):
    if model.v4_fit_ids_sha256 != object_sha(ids):
        raise ValueError("Lexical fit provenance mismatch")
    fresh = build(component).named_steps["vectorizer"]
    with threadpool_limits(limits=4):
        fresh.fit([rows[i]["text"] for i in ids])
    fitted = model.named_steps["vectorizer"]
    if fresh.vocabulary_ != fitted.vocabulary_ or not np.array_equal(fresh.idf_, fitted.idf_):
        raise ValueError("Vocabulary/IDF not reproducible from A_fit alone")
    return {"features": len(fitted.vocabulary_), "fit_ids_sha256": object_sha(ids),
            "idf_and_vocabulary_rebuilt_from_fit_only": True}
