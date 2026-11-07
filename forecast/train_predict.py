"""Финальное обучение на всём train и создание посылки (id, y_pred).

Примеры:
  # одна модель, сумма прогноза приводится к 2.23 млрд
  uv run python -m forecast.train_predict --name v2_l1 --v2 --rounds 900 --target-sum 2.23e9
  # ансамбль из уже обученных моделей (runs/test_<name>.parquet)
  uv run python -m forecast.train_predict --blend v2_l1:0.6 v2_l2:0.4 --target-sum 2.23e9
  # CatBoost
  uv run python -m forecast.train_predict --name f_cb_mae --lib cb --objective MAE --v2 --no-route-cat --rounds 900
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecast.features import HORIZON, STEPS_PER_DAY, FeatureBuilder, feature_columns, load_panel

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def train_one(args: argparse.Namespace) -> pd.DataFrame:
    t0 = time.time()
    panel = load_panel(ROOT / "Data" / "train_solo_track.parquet")
    fb = FeatureBuilder(panel, v2=args.v2)
    last = panel.n_times - 1
    test_slot = last % STEPS_PER_DAY
    slots = range(test_slot - args.slot_radius, test_slot + args.slot_radius + 1)
    tr_anchors = np.sort(np.concatenate([fb.valid_anchors(slot=s) for s in slots]))
    tr = fb.build(tr_anchors)
    te = fb.build(np.array([last]), with_target=False)
    feats = [c for c in feature_columns(tr) if not (args.no_route_cat and c == "route_cat")]
    cats = [c for c in ("dow", "route_cat") if c in feats]
    print(f"train rows={len(tr)} anchors={len(tr_anchors)} feats={len(feats)} ({time.time() - t0:.0f}s)", flush=True)

    preds = []
    if args.lib == "cb":
        from catboost import CatBoostRegressor, Pool

        pool = Pool(tr[feats], tr["target_norm"], weight=tr["scale"], cat_features=["dow"])
        for seed in range(args.seeds):
            model = CatBoostRegressor(
                loss_function=args.objective,
                iterations=args.rounds,
                learning_rate=args.lr,
                depth=8,
                border_count=64,
                random_seed=42 + seed,
                thread_count=-1,
                verbose=0,
            )
            model.fit(pool)
            preds.append(model.predict(te[feats]))
            print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    dtr = None if args.lib == "cb" else lgb.Dataset(
        tr[feats], tr["target_norm"], weight=tr["scale"], categorical_feature=cats, free_raw_data=False
    )
    for seed in range(args.seeds if args.lib == "lgb" else 0):
        params = dict(
            objective=args.objective,
            learning_rate=args.lr,
            num_leaves=63,
            min_data_in_leaf=200,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=1.0,
            max_bin=args.max_bin,
            cat_smooth=50,
            verbose=-1,
            seed=42 + seed,
        )
        if args.objective == "huber":
            params["alpha"] = 0.5
        model = lgb.train(params, dtr, num_boost_round=args.rounds)
        preds.append(model.predict(te[feats]))
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    out = te[["route_id", "k"]].copy()
    out["timestamp"] = panel.times[last] + pd.to_timedelta(out["k"] * 30, unit="min")
    out["pred"] = np.clip(np.mean(preds, axis=0), 0, None) * te["scale"].to_numpy()
    RUNS.mkdir(exist_ok=True)
    out.to_parquet(RUNS / f"test_{args.name}.parquet")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="model")
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--no-route-cat", action="store_true")
    ap.add_argument("--slot-radius", type=int, default=4)
    ap.add_argument("--lib", choices=["lgb", "cb"], default="lgb")
    ap.add_argument("--objective", default="l1", help="lgb: l1/l2/huber; cb: MAE/RMSE")
    ap.add_argument("--rounds", type=int, help="число итераций (по валидации)")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--max-bin", type=int, default=63)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--blend", nargs="+", help="name:weight ... - смешать сохранённые прогнозы вместо обучения")
    ap.add_argument("--mult", type=float, default=1.0, help="множитель (если не задан --target-sum)")
    ap.add_argument("--target-sum", type=float, default=None, help="привести сумму прогноза к этому значению")
    ap.add_argument("--out", default=str(ROOT / "test.csv"))
    args = ap.parse_args()

    if args.blend:
        parts = [(n, float(w)) for n, w in (s.split(":") for s in args.blend)]
        frames = [pd.read_parquet(RUNS / f"test_{n}.parquet") for n, _ in parts]
        out = frames[0][["route_id", "k", "timestamp"]].copy()
        total_w = sum(w for _, w in parts)
        out["pred"] = sum(f["pred"].to_numpy() * w for f, (_, w) in zip(frames, parts, strict=True)) / total_w
    else:
        if args.rounds is None:
            ap.error("--rounds обязателен при обучении")
        out = train_one(args)

    raw_sum = out["pred"].sum()
    mult = args.target_sum / raw_sum if args.target_sum else args.mult
    out["y_pred"] = out["pred"] * mult
    print(f"сумма до масштабирования {raw_sum / 1e9:.4f} млрд, множитель {mult:.4f}")

    test = pd.read_parquet(ROOT / "Data" / "test_solo_track.parquet")
    test["timestamp"] = pd.to_datetime(test["timestamp"])
    sub = (
        test.merge(out[["route_id", "timestamp", "y_pred"]], on=["route_id", "timestamp"], how="left")
        .sort_values("id")[["id", "y_pred"]]
        .reset_index(drop=True)
    )
    assert len(sub) == len(test) == len(out) and sub["id"].is_unique
    assert sub["y_pred"].notna().all(), "не для всех точек теста есть прогноз"
    assert len(out) % HORIZON == 0
    sub.to_csv(args.out, index=False)
    print(f"saved {args.out}: rows={len(sub)} sum={sub['y_pred'].sum() / 1e9:.4f} млрд")


if __name__ == "__main__":
    main()
