"""Fixed-update BERT trainer with complete update-boundary checkpoints."""
import hashlib
import inspect
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from common import BASE, BASE_SHA256, CLASSES, digest
from runtime import FROZEN, atomic_json, environment, implementation, json_read, object_sha, validate_probabilities

SEED = 20260941


def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("V4 BERT requires a verified CUDA GPU")


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def order_for_epoch(n, epoch):
    if epoch < 1 or n < 8:
        raise ValueError("Need epoch >=1 and at least eight fit examples")
    return torch.randperm(n, generator=torch.Generator().manual_seed(SEED + 100003 * epoch)).tolist()[:(n // 8)*8]


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8,
                             amsgrad=False, maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)


def initialize():
    setup()
    if digest(BASE / "pytorch_model.bin") != BASE_SHA256:
        raise ValueError("Original pretrained weights changed")
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, use_fast=True)
    seed_all()
    model, info = AutoModelForSequenceClassification.from_pretrained(
        BASE, local_files_only=True, num_labels=3, label2id={c: i for i, c in enumerate(CLASSES)},
        id2label={i: c for i, c in enumerate(CLASSES)}, output_loading_info=True)
    missing = set(info["missing_keys"])
    if missing != {"classifier.weight", "classifier.bias"} or info.get("mismatched_keys"):
        raise ValueError(f"Unexpected pretrained loading report: {info}")
    cfg = model.config
    if (cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size, cfg.num_labels) != (12, 12, 768, 3):
        raise ValueError("BERT architecture mismatch")
    initialization = {"loading_info": info, "classifier_initial_sha256": state_hash(model.classifier.state_dict()),
                      "original_weights_sha256": BASE_SHA256, "attention_implementation": model.config._attn_implementation,
                      "precision": "fp32; TF32 disabled; AMP disabled"}
    return model.to("cuda:0"), tokenizer, initialization


def checkpoint_save(path, model, optimizer, scheduler, cursor, identity, initialization):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "rng": rng_state(), "cursor": cursor,
                "identity": identity, "initialization": initialization}, tmp)
    os.replace(tmp, path)
    atomic_json(path.with_suffix(".integrity.json"), {"sha256": digest(path), "cursor": cursor, "identity": identity})


def checkpoint_load(path, model, optimizer, scheduler, identity):
    meta = json_read(Path(path).with_suffix(".integrity.json"))
    if meta["sha256"] != digest(path) or meta["identity"] != identity:
        raise ValueError("Checkpoint checksum/provenance mismatch")
    # Trusted local checkpoint only, after hash and identity validation; contains RNG tuples.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["identity"] != identity or state["cursor"] != meta["cursor"]:
        raise ValueError("Checkpoint identity/cursor mismatch")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"])
    return state["cursor"], state["initialization"]


def train(rows, ids, directory, total_updates=3557, stop_at=None, checkpoint_every=128):
    """stop_at permits a diagnostic interruption; total duration remains part of identity."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model, tokenizer, initialization = initialize()
    tokens = tokenizer([rows[i]["text"] for i in ids], truncation=True, max_length=512, padding=False)
    labels = [CLASSES.index(rows[i]["label"]) for i in ids]
    collate = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    optimizer = optimizer_for(model)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=470, num_training_steps=4702)
    identity = {"fit_ids_sha256": object_sha(ids), "training_sha256": json_read(FROZEN / "protocol.json")["training_sha256"],
                "protocol_sha256": digest(FROZEN / "protocol.json"), "total_updates": total_updates,
                "environment": environment(), "implementation_sha256": object_sha(implementation())}
    cursor = {"epoch": 1, "update_in_epoch": 0, "completed_updates": 0}
    checkpoint = directory / "resume.pt"
    resumed = checkpoint.exists()
    if resumed:
        cursor, initialization = checkpoint_load(checkpoint, model, optimizer, scheduler, identity)
    # Prime previously encountered shapes in the new CUDA context, then discard all
    # effects on parameters, gradients, optimizer, scheduler and RNG by restoring again.
    priming_started = time.perf_counter()
    primed_shapes = set()
    if resumed and cursor["completed_updates"]:
        model.train()
        for epoch in range(1, cursor["epoch"] + 1):
            order = order_for_epoch(len(ids), epoch)
            consumed = len(order) if epoch < cursor["epoch"] else cursor["update_in_epoch"] * 8
            for start in range(0, consumed, 2):
                indices = order[start:start+2]
                shape = max(len(tokens["input_ids"][j]) for j in indices)
                if shape in primed_shapes:
                    continue
                primed_shapes.add(shape)
                records = [{**{k: v[j] for k, v in tokens.items()}, "labels": labels[j]} for j in indices]
                batch = {k: v.to("cuda:0") for k, v in collate(records).items()}
                (model(**batch).loss / 4).backward()
                optimizer.zero_grad(set_to_none=True)
        cursor, initialization = checkpoint_load(checkpoint, model, optimizer, scheduler, identity)
    priming_seconds = time.perf_counter() - priming_started
    limit = total_updates if stop_at is None else min(stop_at, total_updates)
    if not 0 <= cursor["completed_updates"] <= limit or len(ids) < 8:
        raise ValueError("Invalid training cursor/fit size")
    torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    initial_step = cursor["completed_updates"]
    timings, losses, lr_trace = [], [], []
    while cursor["completed_updates"] < limit:
        order = order_for_epoch(len(ids), cursor["epoch"])
        per_epoch = len(order) // 8
        for u in range(cursor["update_in_epoch"], per_epoch):
            step_started = time.perf_counter()
            loss_sum = 0.0
            lr_trace.append(optimizer.param_groups[0]["lr"])
            for micro in range(4):
                indices = order[u*8 + micro*2:u*8 + micro*2 + 2]
                records = [{**{k: v[j] for k, v in tokens.items()}, "labels": labels[j]} for j in indices]
                batch = {k: v.to("cuda:0") for k, v in collate(records).items()}
                loss = model(**batch).loss / 4
                loss.backward()
                loss_sum += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            cursor["completed_updates"] += 1
            cursor["update_in_epoch"] = u + 1
            if cursor["update_in_epoch"] == per_epoch:
                cursor["epoch"] += 1
                cursor["update_in_epoch"] = 0
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - step_started)
            losses.append(loss_sum)
            if cursor["completed_updates"] % checkpoint_every == 0 or cursor["completed_updates"] == limit:
                checkpoint_save(checkpoint, model, optimizer, scheduler, dict(cursor), identity, initialization)
                print(f"BERT updates {cursor['completed_updates']}/{total_updates}; epoch={cursor['epoch']}; loss={loss_sum:.6f}", flush=True)
            if cursor["completed_updates"] >= limit:
                break
    result = {"cursor": cursor, "resumed": resumed, "executed_updates": cursor["completed_updates"] - initial_step,
              "wall_seconds": time.perf_counter() - started, "update_seconds": timings, "losses": losses,
              "discarded_backend_priming_microbatches": len(primed_shapes), "backend_priming_seconds": priming_seconds,
              "lr_before_updates": lr_trace, "lr_next_update": optimizer.param_groups[0]["lr"],
              "scheduler_last_epoch": scheduler.last_epoch,
              "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved(), "initialization": initialization,
              "model_sha256": state_hash(model.state_dict()), "identity": identity}
    log_path = directory / f"training_to_{cursor['completed_updates']:05d}.json"
    if log_path.exists():
        log_path = directory / f"training_to_{cursor['completed_updates']:05d}_attempt_{time.time_ns()}.json"
    atomic_json(log_path, result)
    if cursor["completed_updates"] == total_updates:
        final = directory / "model"
        model.save_pretrained(final, safe_serialization=True)
        tokenizer.save_pretrained(final)
    return model, tokenizer, result


def predict(model, tokenizer, texts, batch_size=4):
    model.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(texts[start:start+batch_size], max_length=512, truncation=True, padding=True, return_tensors="pt")
            out.append(torch.softmax(model(**{k: v.to("cuda:0") for k, v in tokens.items()}).logits, dim=-1).cpu().numpy())
    return validate_probabilities(np.concatenate(out) if out else np.empty((0, 3)), len(texts))


def reload_model(path):
    setup()
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to("cuda:0")
    if [model.config.id2label[i] for i in range(3)] != list(CLASSES):
        raise ValueError("Reloaded BERT class order mismatch")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model, tokenizer
