"""V4 source and training-data contracts. No imports from earlier implementations."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "training"
CLASSES = ("c1", "c234", "c5")
TRAIN_SHA256 = "0e9219233664ac675fc47982bbc822acd3bb15d67fb7bfc26b67bec047e46818"
BASE_SHA256 = "cb1693767adef60abf23d9fde3996f0c1e6310afad103a2db94ad44854568955"
REVISION = "94d69c95f98f7d5b2a8700c420230ae10def0baa"
EP1_COMMIT = "596f81606913d13d3d51910fa49b4710ce261421"
BASE = ROOT / ".cache/bertimbau"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a prior audit run or protocol silently.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def group_id(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def read_train():
    """Only the official training path is exposed; no arbitrary XLSX input."""
    import openpyxl
    path = ROOT / "data/train.xlsx"
    if digest(path) != TRAIN_SHA256:
        raise ValueError("Official train.xlsx SHA-256 mismatch")
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=False)
    try:
        sheet = workbook["train"]
        values = sheet.iter_rows()
        if tuple(cell.value for cell in next(values)) != ("resp_text", "clarity"):
            raise ValueError("Expected exactly resp_text, clarity")
        result = []
        for row_id, cells in enumerate(values):
            text, label = cells
            if text.data_type == "f" or label.data_type == "f":
                raise ValueError("Formula cells require explicit review")
            if label.value not in CLASSES:
                raise ValueError(f"Unexpected label at source row_id={row_id}")
            result.append({"row_id": row_id, "text": str(text.value),
                           "label": label.value, "cell_type": type(text.value).__name__,
                           "group": group_id(str(text.value))})
        if len(result) != 20092:
            raise ValueError("Expected 20092 source rows")
        return result
    finally:
        workbook.close()


def deduplicate_fit(rows):
    seen = set()
    kept = []
    for row in rows:
        key = (row["text"], row["label"])
        if key not in seen:
            kept.append(row)
            seen.add(key)
    return kept


def redundancy(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {
        "groups": len(groups),
        "duplicate_groups": sum(len(v) > 1 for v in groups.values()),
        "rows_in_duplicate_groups": sum(len(v) for v in groups.values() if len(v) > 1),
        "redundant_rows_ignoring_label": len(rows) - len(groups),
        "conflicting_groups": sum(len({r['label'] for r in v}) > 1 for v in groups.values()),
        "rows_in_conflicting_groups": sum(len(v) for v in groups.values() if len({r['label'] for r in v}) > 1),
        "largest_group": max(map(len, groups.values()), default=0),
    }
