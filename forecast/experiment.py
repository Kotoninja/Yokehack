"""Быстрые эксперименты: кэш признаков + несколько вариантов модели на одной валидации.

Главная метрика сравнения - score_sum: скор после того, как сумма прогноза за каждый
валидационный день приведена к фактической. Общий уровень на тесте мы калибруем по
лидерборду отдельно, поэтому модели сравниваем по тому, как они распределяют объём
между маршрутами и часами.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecast.features import FeatureBuilder, feature_columns, load_panel, score

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
TEST_SLOT = 21

V2_ONLY_PREFIXES = ("today_vs", "g_today", "shape_fc", "s4_over", "s123_", "route_cat")
V2_ONLY_SUFFIXES = ("_d1h", "_d1d", "_today")


def is_v2_only(col: str) -> bool:
    return col.startswith(V2_ONLY_PREFIXES) or col.endswith(V2_ONLY_SUFFIXES)


def load_or_build(radius: int, val_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    RUNS.mkdir(exist_ok=True)
    ptr, pva = RUNS / f"feat_r{radius}_tr.parquet", RUNS / f"feat_r{radius}_va.parquet"
    if ptr.exists() and pva.exists():
        return pd.read_parquet(ptr), pd.read_parquet(pva)
    panel = load_panel(ROOT / "Data" / "train_solo_track.parquet")
    fb = FeatureBuilder(panel, v2=True)
    val_anchors = fb.valid_anchors(slot=TEST_SLOT)[-val_days:]
    slots = range(TEST_SLOT - radius, TEST_SLOT + radius + 1)
    tr_anchors = np.sort(np.concatenate([fb.valid_anchors(slot=s) for s in slots]))
    tr_anchors = tr_anchors[tr_anchors + 8 <= val_anchors[0]]
    tr, va = fb.build(tr_anchors), fb.build(val_anchors)
    for df in (tr, va):
        num = df.select_dtypes("float64").columns
        df[num] = df[num].astype(np.float32)
    tr.to_parquet(ptr)
    va.to_parquet(pva)
    return tr, va


def sum_matched(va: pd.DataFrame, col: str) -> float:
    """Скор, когда сумма прогноза за каждый день совпадает с фактом."""
    k = va.groupby("anchor")["target"].transform("sum") / va.groupby("anchor")[col].transform("sum")
    return score(va["target"], va[col] * k)


VARIANTS = {
    "v1_l1": dict(fs="v1", objective="l1"),
    "v2_l1": dict(fs="v2", objective="l1"),
    "v2_l2": dict(fs="v2", objective="l2"),
    "v2_huber": dict(fs="v2", objective="huber", alpha=0.5),
    "v2_l1_s1": dict(fs="v2", objective="l1", seed=1),
    "v2_l1_s2": dict(fs="v2", objective="l1", seed=2),
    "v2_l1_nocat": dict(fs="v2_nocat", objective="l1"),
    "v2_l2_nocat": dict(fs="v2_nocat", objective="l2"),
    "v2_huber_nocat": dict(fs="v2_nocat", objective="huber", alpha=0.5),
    "v1_l2": dict(fs="v1", objective="l2"),
    "cb_rmse": dict(fs="v2_nocat", lib="cb", loss="RMSE"),
    "cb_mae": dict(fs="v2_nocat", lib="cb", loss="MAE"),
}


def fit_catboost(cfg: dict, tr: pd.DataFrame, va: pd.DataFrame, feats: list[str], w: np.ndarray, seed: int):
    from catboost import CatBoostRegressor, Pool

    model = CatBoostRegressor(
        loss_function=cfg["loss"],
        eval_metric="MAE",
        iterations=5000,
        learning_rate=0.1,
        depth=8,
        border_count=64,
        random_seed=42 + seed,
        od_type="Iter",
        od_wait=100,
        thread_count=-1,
        verbose=0,
    )
    model.fit(
        Pool(tr[feats], tr["target_norm"], weight=w, cat_features=["dow"]),
        eval_set=Pool(va[feats], va["target_norm"], weight=va["scale"], cat_features=["dow"]),
        use_best_model=True,
    )
    return model.predict(va[feats]), model.get_best_iteration()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+", choices=list(VARIANTS))
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--val-days", type=int, default=14)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--decay-days", type=float, default=None, help="вес якоря exp(-возраст/decay), возраст в днях")
    ap.add_argument("--tag", default="", help="суффикс к имени прогона")
    args = ap.parse_args()

    t0 = time.time()
    tr, va = load_or_build(args.radius, args.val_days)
    all_feats = feature_columns(tr)
    print(f"rows train={len(tr)} val={len(va)} ({time.time() - t0:.0f}s)", flush=True)

    for variant in args.variants:
        name = variant + args.tag
        cfg = dict(VARIANTS[variant])
        fs = cfg.pop("fs")
        seed = cfg.pop("seed", 0)
        if fs == "v1":
            feats = [c for c in all_feats if not is_v2_only(c)]
        elif fs == "v2_nocat":
            feats = [c for c in all_feats if c != "route_cat"]
        else:
            feats = all_feats
        cats = [c for c in ("dow", "route_cat") if c in feats]
        w = tr["scale"].to_numpy()
        if args.decay_days:
            age_days = (tr["anchor"].max() - tr["anchor"].to_numpy()) / 48
            w = w * np.exp(-age_days / args.decay_days)
        t1 = time.time()
        if cfg.get("lib") == "cb":
            raw_pred, best_iter = fit_catboost(cfg, tr, va, feats, w, seed)
        else:
            raw_pred, best_iter = fit_lgb(cfg, args.lr, seed, tr, va, feats, cats, w)
        pred = np.clip(raw_pred, 0, None) * va["scale"]
        out = va[["route_id", "anchor", "k", "target"]].copy()
        out["pred"] = pred
        out.to_parquet(RUNS / f"val_{name}.parquet")
        res = dict(
            name=name,
            raw=round(score(out["target"], out["pred"]), 4),
            sum_matched=round(sum_matched(out, "pred"), 4),
            best_iter=best_iter,
            n_feats=len(feats),
            minutes=round((time.time() - t1) / 60, 1),
        )
        print(json.dumps(res, ensure_ascii=False), flush=True)
        with open(RUNS / "results.jsonl", "a") as fh:
            fh.write(json.dumps({**res, "radius": args.radius, "lr": args.lr, "decay": args.decay_days}) + "\n")


def fit_lgb(cfg: dict, lr: float, seed: int, tr, va, feats: list[str], cats: list[str], w: np.ndarray):
    params = dict(
        learning_rate=lr,
        num_leaves=63,
        min_data_in_leaf=200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        max_bin=63,
        cat_smooth=50,
        verbose=-1,
        seed=42 + seed,
        metric="l1",  # ранняя остановка по взвешенному L1 (== WAPE) независимо от целевой функции
        **cfg,
    )
    dtr = lgb.Dataset(tr[feats], tr["target_norm"], weight=w, categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(va[feats], va["target_norm"], weight=va["scale"], reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=5000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    return model.predict(va[feats], num_iteration=model.best_iteration), model.best_iteration


if __name__ == "__main__":
    main()
