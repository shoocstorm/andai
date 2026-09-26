# Goldens for src-tauri laya parity tests: laya-mlx FP32 on tests/fixtures/laya/fixture.json.
# Run with the laya-mlx venv: python golden.py fixture.json <checkpoint dir> <label> golden-<id>.json
import json, sys, numpy as np
import laya_mlx as laya
from laya_mlx.agent import collate_items
fx = json.load(open(sys.argv[1]))
qs = {"next": {"type": "choice", "instructions": fx["question"], "criteria": dict(fx["options"])}}
a = laya.load(sys.argv[2], dtype="float32")
items, _ = a.prepare(fx["state"], qs)
logits, _ = a.forward(collate_items(items, a.tok.pad_token_id))
k = len(items[0]["markers"])
bucket = "choice:" + ("2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+")
t = a.temperature_by_options.get(bucket, a.temperature[0])
z = np.asarray(logits, dtype=np.float64)[0, :k] / t
p = np.exp(z - z.max()); p /= p.sum()
json.dump({"source": "laya-mlx 0.2.0 (" + sys.argv[3] + ", float32)", "ids": items[0]["ids"], "markers": items[0]["markers"], "probabilities_fp32": p.tolist()}, open(sys.argv[4], "w"), indent=1)
print(sys.argv[3], len(items[0]["ids"]), np.round(p, 4))
