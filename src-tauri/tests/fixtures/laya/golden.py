# Goldens for src-tauri laya parity tests: laya-mlx FP32 on tests/fixtures/laya/fixture.json.
# Run with the laya-mlx venv: python golden.py fixture.json <checkpoint dir> <label> golden-<id>.json
import json, sys, numpy as np
import laya_mlx as laya
from laya_mlx.agent import collate_items
fx = json.load(open(sys.argv[1]))
choice = {"type": "choice", "instructions": fx["question"], "criteria": dict(fx["options"])}
stop = {"type": "noul", "instructions": fx["stop"]}
a = laya.load(sys.argv[2], dtype="float32")

def probs(questions):
    """Calibrated probabilities per question, from one padded batch (laya-mlx collate)."""
    items, internal = a.prepare(fx["state"], questions)
    logits, _ = a.forward(collate_items(items, a.tok.pad_token_id))
    out = []
    for row, (item, q) in enumerate(zip(items, internal)):
        k = len(item["markers"])
        size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
        t = a.temperature_by_options.get(q["t"] + ":" + size, a.temperature[item["qtype"]])
        z = np.asarray(logits, dtype=np.float64)[row, :k] / t
        p = np.exp(z - z.max())
        out.append((p / p.sum()).tolist())
    return items, out

items, [single] = probs({"next": choice})
_, [b_choice, b_stop] = probs({"next": choice, "stop": stop})
rel = fx["relevance"]
rel_states = [f"User request:\n{rel['request']}\n\nPassage from {src}:\n{text}" for src, text in rel["passages"]]
# One noul row per passage, each with its own state, in one padded batch.
rel_items = []
for st in rel_states:
    it, _ = a.prepare(st, {"rel": {"type": "noul", "instructions": rel["statement"]}})
    rel_items += it
rel_logits, _ = a.forward(collate_items(rel_items, a.tok.pad_token_id))
t_noul = a.temperature_by_options.get("noul:2", a.temperature[2])
relevance = []
for row in range(len(rel_items)):
    z = np.asarray(rel_logits, dtype=np.float64)[row, :2] / t_noul
    q = np.exp(z - z.max())
    relevance.append(float((q / q.sum())[1]))

json.dump(
    {
        "source": "laya-mlx 0.2.0 (" + sys.argv[3] + ", float32)",
        "ids": items[0]["ids"],
        "markers": items[0]["markers"],
        "probabilities_fp32": single,
        "batch": {"choice_fp32": b_choice, "stop_fp32": b_stop},
        "relevance_fp32": relevance,
    },
    open(sys.argv[4], "w"),
    indent=1,
)
print(sys.argv[3], len(items[0]["ids"]), np.round(single, 4), "stop P(true)", round(b_stop[1], 4), "relevance", np.round(relevance, 4))
