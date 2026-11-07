"""Selection and calibration use held-out B predictions only; E never chooses parameters."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from threadpoolctl import threadpool_limits
from common import CLASSES
from runtime import FROZEN, json_read, validate_probabilities

ORDER = ("word", "char", "bert", "original_equal", "weighted_raw", "mean_three", "weighted_calibrated", "stack_two", "stack_three")
ALPHAS = (0.5, 0.75, 0.25, 1.0, 0.0)


def score(y, p):
    prediction = np.asarray(CLASSES)[np.argmax(p, axis=1)]
    return float(accuracy_score(y, prediction)), float(f1_score(y, prediction, labels=list(CLASSES), average="macro", zero_division=0))


def best(scores, order, margin=0):
    winner = order[0]
    for name in order[1:]:
        a, f = scores[name]
        wa, wf = scores[winner]
        if a > wa + 1e-12 or (abs(a-wa) <= 1e-12 and f > wf + 1e-12):
            winner = name
    if margin:
        return next(name for name in order if scores[name][0] >= scores[winner][0] - margin - 1e-12)
    return winner


def temperature(p, t):
    return softmax(np.log(np.clip(p, 1e-12, 1)) / t, axis=1)


def fit_temperature(p, y):
    result = minimize_scalar(lambda z: log_loss(y, temperature(p, np.exp(z)), labels=list(CLASSES)),
                             bounds=(np.log(0.05), np.log(20)), method="bounded",
                             options={"xatol": 1e-6, "maxiter": 200})
    if not result.success:
        raise RuntimeError("Temperature optimization failed")
    return float(np.exp(result.x))


def stacker():
    return LogisticRegression(penalty="l2", C=1, solver="lbfgs", max_iter=1000, tol=1e-4,
                              fit_intercept=True, random_state=20261027, n_jobs=1, dual=False,
                              class_weight=None, warm_start=False, l1_ratio=None, verbose=0)


def features(p, three):
    return np.concatenate([p[c] for c in (("bert", "word", "char") if three else ("bert", "word"))], axis=1)


def fixed(p):
    return {"bert": p["bert"], "word": p["word"], "char": p["char"],
            "original_equal": (p["bert"] + p["word"]) / 2,
            "mean_three": (p["bert"] + p["word"] + p["char"]) / 3}


def select_and_fit(p, y, ids, inner):
    """Return combiners and internal held-out scores; no external predictions accepted."""
    y = np.asarray(y)
    p = {c: validate_probabilities(p[c], len(ids)) for c in ("bert", "word", "char")}
    position = {row: i for i, row in enumerate(ids)}
    if len(position) != len(ids):
        raise ValueError("Duplicate B IDs")
    raw = {a: a*p["bert"] + (1-a)*p["word"] for a in ALPHAS}
    cal = {a: np.zeros_like(p["bert"]) for a in ALPHAS}
    oof = fixed(p)
    oof.update({"stack_two": np.zeros_like(p["bert"]), "stack_three": np.zeros_like(p["bert"])})
    coverage = np.zeros(len(ids), dtype=int)
    for split in inner:
        tr = np.array([position[i] for i in split["train"]])
        va = np.array([position[i] for i in split["validation"]])
        if set(tr) & set(va) or set(tr) | set(va) != set(range(len(ids))):
            raise ValueError("Invalid internal B indices")
        coverage[va] += 1
        ts = {c: fit_temperature(p[c][tr], y[tr]) for c in ("bert", "word")}
        for a in ALPHAS:
            cal[a][va] = a*temperature(p["bert"][va], ts["bert"]) + (1-a)*temperature(p["word"][va], ts["word"])
        for name in ("stack_two", "stack_three"):
            x = features(p, name == "stack_three")
            with threadpool_limits(limits=4):
                model = stacker().fit(x[tr], y[tr])
                oof[name][va] = model.predict_proba(x[va])
    if not np.all(coverage == 1):
        raise ValueError("Internal B validation must cover every row exactly once")
    alpha_raw = best({a: score(y, raw[a]) for a in ALPHAS}, ALPHAS)
    alpha_cal = best({a: score(y, cal[a]) for a in ALPHAS}, ALPHAS)
    oof["weighted_raw"], oof["weighted_calibrated"] = raw[alpha_raw], cal[alpha_cal]
    scores = {name: score(y, oof[name]) for name in ORDER}
    selected = best(scores, ORDER, margin=0.005)
    models = {}
    with threadpool_limits(limits=4):
        for name in ("stack_two", "stack_three"):
            models[name] = stacker().fit(features(p, name == "stack_three"), y)
    return {"alpha_raw": alpha_raw, "alpha_calibrated": alpha_cal,
            "temperatures": {c: fit_temperature(p[c], y) for c in ("bert", "word")},
            "stackers": models, "selected": selected,
            "internal_scores": {name: {"accuracy": s[0], "macro_f1": s[1]} for name, s in scores.items()}}


def apply(combiners, p):
    out = fixed(p)
    a = combiners["alpha_raw"]
    out["weighted_raw"] = a*p["bert"] + (1-a)*p["word"]
    a = combiners["alpha_calibrated"]
    ts = combiners["temperatures"]
    out["weighted_calibrated"] = a*temperature(p["bert"], ts["bert"]) + (1-a)*temperature(p["word"], ts["word"])
    with threadpool_limits(limits=4):
        for name in ("stack_two", "stack_three"):
            out[name] = combiners["stackers"][name].predict_proba(features(p, name == "stack_three"))
    out["selected_pipeline"] = out[combiners["selected"]]
    return {name: validate_probabilities(value) for name, value in out.items()}
