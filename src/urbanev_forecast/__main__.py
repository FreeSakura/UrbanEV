"""Train a development model or evaluate a checkpoint using current code."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
from .data import fold_bounds, load_rate_prefix, scores, window_starts


def code_fingerprint():
    root=Path(__file__).resolve().parents[2]
    paths=sorted((root/"src/urbanev_forecast").glob("*.py"))+sorted((root/"models/timexer").rglob("*.py"))
    digest=hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode()+b"\0")
        digest.update(path.read_bytes().replace(b"\r\n",b"\n"))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(prog="python -m urbanev_forecast")
    parser.add_argument("command", choices=("smoke", "train", "test"))
    parser.add_argument("--csv", type=Path, help="Prepared hourly occupancy-rate CSV, not raw counts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", choices=("seasonal_linear", "seasonal_mlp", "innovation_attention", "level_attention", "timexer"), default="seasonal_linear")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--horizon", type=int, choices=(3,6,9,12), default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists() and not args.output.is_dir():
        parser.error("output must be a directory")
    output_names = ("result.json",) if args.command == "test" else ("result.json", "checkpoint.pt")
    if any((args.output / name).exists() for name in output_names):
        parser.error("Output artifacts already exist; choose another output directory")
    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("epochs/patience/batch-size must be positive")
    import torch
    from torch.utils.data import Dataset, DataLoader
    from .models import build_model
    torch.set_num_threads(2)
    started = time.perf_counter()
    current_code = code_fingerprint()
    frozen = None
    if args.command == "test":
        if args.checkpoint is None or args.csv is None:
            parser.error("test requires --checkpoint and --csv")
        frozen = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = frozen["config"]
        if frozen["source_kind"] != "urbanev_prepared_rate":
            parser.error("Synthetic checkpoints cannot be used for benchmark tests")
    else:
        config = dict(model=args.model, history=168, horizon=args.horizon, channels=275,
                      width=64, seed=args.seed, fold=args.fold)
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable")
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = False
    if args.command == "smoke":
        config["channels"] = 8
        rng = np.random.default_rng(config["seed"])
        t = np.arange(480)[:,None]
        values = np.clip(.4+.15*np.sin(t*2*np.pi/24+np.arange(8)[None,:]/8)+rng.normal(0,.02,(480,8)),0,1).astype(np.float32)
        train_end, val_end = 360, 480
        source = {"source_kind":"synthetic_only", "seed":config["seed"]}
    else:
        if args.csv is None:
            parser.error("train requires --csv")
        train_end, val_end, test_end = fold_bounds(config["fold"])
        values, source = load_rate_prefix(args.csv, test_end if args.command == "test" else val_end)
        source["source_kind"] = "urbanev_prepared_rate"
    history, horizon = config["history"], config["horizon"]
    class Windows(Dataset):
        def __init__(self, start, stop):
            self.starts = window_starts(len(values), history, horizon, start, stop)
        def __len__(self):return len(self.starts)
        def __getitem__(self, index):
            s = self.starts[index]
            return torch.from_numpy(values[s-history:s]), torch.from_numpy(values[s:s+horizon])
    model = build_model(config["model"], history, horizon, config["channels"], config["width"]).to(args.device)
    def predict(loader):
        model.eval(); ps=[]; ys=[]
        with torch.no_grad():
            for x,y in loader:
                ps.append(model(x.to(args.device)).cpu().numpy()); ys.append(y.numpy())
        return np.concatenate(ps), np.concatenate(ys)
    args.output.mkdir(parents=True, exist_ok=True)
    if frozen is not None:
        # Verify the training/validation prefix before using an immutable checkpoint.
        _, prefix = load_rate_prefix(args.csv, val_end)
        if prefix["prefix_sha256"] != frozen["source"]["prefix_sha256"]:
            raise ValueError("Training/validation data differs from the frozen checkpoint")
        model.load_state_dict(frozen["state_dict"])
        p,y = predict(DataLoader(Windows(val_end,test_end), batch_size=args.batch_size))
        report = {"stage":"benchmark_test", "checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  "checkpoint_code_sha256":frozen["code_sha256"],
                  "evaluation_code_matches_training":frozen["code_sha256"] == current_code}
    else:
        train = DataLoader(Windows(0,train_end), batch_size=args.batch_size, shuffle=True,
                           generator=torch.Generator().manual_seed(config["seed"]))
        val = DataLoader(Windows(train_end,val_end), batch_size=args.batch_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, stale, log, best_state, best_epoch = float("inf"),0,[],None,0
        for epoch in range(1, args.epochs+1):
            model.train()
            for x,y in train:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean((model(x.to(args.device))-y.to(args.device))**2)
                if not torch.isfinite(loss):raise ValueError("Nonfinite training loss")
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            p,y = predict(val); metric = scores(np.clip(p,0,1),y)["rmse"]
            log.append({"epoch":epoch,"validation_rmse":metric})
            if metric < best:
                best, stale, best_epoch = metric,0,epoch
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale += 1
            if stale >= args.patience:break
        model.load_state_dict(best_state);p,y = predict(val)
        torch.save({"config":config,"state_dict":best_state,"source":source,"source_kind":source["source_kind"],"code_sha256":current_code},args.output/"checkpoint.pt")
        report = {"stage":"synthetic_smoke" if args.command=="smoke" else "development_validation", "best_epoch":best_epoch,"training_log":log,
                  "training_recipe":{"optimizer":"AdamW","lr":.001,"weight_decay":.0001,"epochs_cap":args.epochs,"patience":args.patience,"batch_size":args.batch_size}}
    report.update({"config":config,"source":source,"raw":scores(p,y),"clipped":scores(np.clip(p,0,1),y),
                   "score_scope":"all forecast timesteps, origins and channels", "windows":len(p),
                   "parameters":sum(v.numel() for v in model.parameters()),"elapsed_seconds":time.perf_counter()-started,
                   "code_sha256":current_code,"torch_version":str(torch.__version__),"numpy_version":np.__version__,
                   "device":args.device,"sota_claim":False,"real_test_executed":args.command=="test"})
    (args.output/"result.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("stage","raw","clipped","parameters","sota_claim")},indent=2))


if __name__ == "__main__":main()
