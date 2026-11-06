"""Валидация: окна, повторяющие тест (якорь 10:30, прогноз 11:00-14:30), за последние дни train."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecast.features import FeatureBuilder, feature_columns, load_panel, score

ROOT = Path(__file__).resolve().parents[1]
TEST_SLOT = 21  # 10:30 - последняя известная точка перед тестом


def train_anchor_set(fb: FeatureBuilder, slots: list[int], end: int, days: int | None) -> np.ndarray:
    anchors = np.concatenate([fb.valid_anchors(slot=s) for s in slots])
    anchors = anchors[anchors + 8 <= end]
    if days is not None:
        anchors = anchors[anchors > end - days * 48]
    return np.sort(anchors)


def per_day_report(df: pd.DataFrame, pred_col: str) -> pd.Series:
    return df.groupby("anchor").apply(lambda g: score(g["target"], g[pred_col]), include_groups=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-days", type=int, default=14)
    ap.add_argument("--slot-radius", type=int, default=0, help="брать якоря в слотах 21±r")
    ap.add_argument("--train-days", type=int, default=None)
    ap.add_argument("--objective", default="l1")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--save", default=None, help="сохранить валидационные прогнозы (parquet)")
    args = ap.parse_args()

    t0 = time.time()
    panel = load_panel(ROOT / "Data" / "train_solo_track.parquet")
    fb = FeatureBuilder(panel)

    val_anchors = fb.valid_anchors(slot=TEST_SLOT)[-args.val_days :]
    slots = list(range(TEST_SLOT - args.slot_radius, TEST_SLOT + args.slot_radius + 1))
    tr_anchors = train_anchor_set(fb, slots, val_anchors[0], args.train_days)
    print(f"train anchors: {len(tr_anchors)}, val anchors: {len(val_anchors)}")

    tr = fb.build(tr_anchors)
    va = fb.build(val_anchors)
    feats = feature_columns(tr)
    print(f"rows train={len(tr)} val={len(va)} features={len(feats)} ({time.time() - t0:.0f}s)")

    # --- наивные бейзлайны ---
    base = {
        "same_time_yesterday": va["y_d1"] * va["scale"],
        "seasonal_7d_mean": va["seas_7d"] * va["scale"],
        "seasonal_28d_mean": va["seas_28d"] * va["scale"],
        "level_7d": va["scale"] - 1,
        "level_1d": va["lvl_1d"] * va["scale"],
    }
    for name, pred in base.items():
        va[name] = pred
        print(f"{name:>22}: {score(va['target'], pred):.4f}")

    # --- LightGBM на нормированном таргете, вес = масштаб маршрута (L1 == WAPE) ---
    params = dict(
        objective=args.objective,
        learning_rate=args.lr,
        num_leaves=args.leaves,
        min_data_in_leaf=200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        verbose=-1,
        seed=42,
    )
    dtr = lgb.Dataset(tr[feats], tr["target_norm"], weight=tr["scale"], categorical_feature=["dow"])
    dva = lgb.Dataset(va[feats], va["target_norm"], weight=va["scale"], reference=dtr)
    model = lgb.train(
        params,
        dtr,
        num_boost_round=args.rounds,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(500)],
    )
    va["lgb"] = np.clip(model.predict(va[feats], num_iteration=model.best_iteration), 0, None) * va["scale"]
    print(f"{'lgb':>22}: {score(va['target'], va['lgb']):.4f}  best_iter={model.best_iteration}")

    ratio = va["target"].sum() / va["lgb"].sum()
    print(f"sum(y)/sum(pred) на валидации: {ratio:.4f}")
    mults = {m: score(va["target"], va["lgb"] * m) for m in np.arange(0.98, 1.081, 0.01)}
    print("score x mult:", {round(m, 2): round(s, 4) for m, s in mults.items()})
    if args.save:
        va[["route_id", "anchor", "k", "target", "lgb"]].to_parquet(args.save)

    days = pd.DataFrame({c: per_day_report(va, c) for c in ["seasonal_7d_mean", "lgb"]})
    days.index = panel.times[days.index]
    days["opt_mult"] = va.groupby("anchor").apply(
        lambda g: g["target"].sum() / g["lgb"].sum(), include_groups=False
    ).to_numpy()
    print(days.round(4).to_string())
    print("score by k:", va.groupby("k").apply(lambda g: score(g["target"], g["lgb"]), include_groups=False).round(4).to_dict())

    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print((imp / imp.sum()).head(20).round(4).to_string())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
