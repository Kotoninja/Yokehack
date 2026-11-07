"""Сквозной пайплайн: данные -> test.csv. Никаких параметров с лидерборда.

Этапы:
  1. валидация: каждая модель ансамбля обучается на якорях до последних 14 дней,
     ранняя остановка по взвешенному L1 на валидационных окнах (10:30 -> 11:00-14:30);
  2. веса ансамбля подбираются на валидации по sum_matched;
  3. уровень: множитель sum(факт) / sum(ансамбль) на валидационных днях
     (или оценка по соотношению «утро -> после обеда», флаг --level);
  4. финал: каждая модель переобучается на всех якорях с числом итераций из этапа 1,
     прогноз от последней точки train, смешивание, множитель, проверки, CSV.

Запуск:
  uv run python -m forecast.pipeline                 # всё целиком (~1 ч на ноутбуке)
  uv run python -m forecast.pipeline --reuse-validation   # только финал, конфиг из runs/pipeline.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from forecast.experiment import fit_catboost, fit_lgb, is_v3_only, load_or_build, sum_matched
from forecast.features import (
    HORIZON,
    STEPS_PER_DAY,
    FeatureBuilder,
    feature_columns,
    load_panel,
    score,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
TEST_SLOT = 21
RADIUS = 4  # якоря 8:30-12:30
VAL_DAYS = 14
LR = 0.1
ITER_MARGIN = 1.2  # на полных данных строк на ~25% больше -> итераций с запасом

# Состав ансамбля: признаки v3 (кроме CatBoost - v2), разные цели для разнообразия ошибок.
MODELS = {
    "huber_v3": dict(lib="lgb", fs="v3", objective="huber", alpha=0.5, feature_fraction=0.5, bagging_fraction=0.7),
    "q52_v3": dict(lib="lgb", fs="v3", objective="quantile", alpha=0.52),
    "l2_v3": dict(lib="lgb", fs="v3", objective="l2"),
    "cb_mae_v2": dict(lib="cb", fs="v2", loss="MAE"),
}
LGB_BASE = dict(num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8)


def select_features(cols: list[str], fs: str) -> list[str]:
    cols = [c for c in cols if c != "route_cat"]
    return cols if fs == "v3" else [c for c in cols if not is_v3_only(c)]


def lgb_cfg(spec: dict) -> dict:
    cfg = {**LGB_BASE, **{k: v for k, v in spec.items() if k not in ("lib", "fs")}}
    if cfg["objective"] == "l2":
        cfg.pop("alpha", None)
    return cfg


# ---------------------------------------------------------------- этап 1-3
def validate(level_mode: str) -> dict:
    t0 = time.time()
    tr, va = load_or_build(RADIUS, VAL_DAYS, "v3")
    cols = feature_columns(tr)
    w = tr["scale"].to_numpy()
    preds, iters = {}, {}
    for name, spec in MODELS.items():
        feats = select_features(cols, spec["fs"])
        if spec["lib"] == "cb":
            raw, it = fit_catboost(spec, tr, va, feats, w, seed=0)
        else:
            raw, it = fit_lgb(lgb_cfg(spec), LR, 0, tr, va, feats, ["dow"], w)
        preds[name] = np.clip(raw, 0, None) * va["scale"].to_numpy()
        iters[name] = int(it)
        out = va[["anchor", "target"]].assign(p=preds[name])
        print(f"[val] {name:10s} sum_matched={sum_matched(out, 'p'):.4f} iter={it} ({time.time() - t0:.0f}s)", flush=True)

    base = va[["anchor", "target"]].copy()
    p = np.column_stack([preds[n] for n in MODELS])

    def obj(x: np.ndarray) -> float:
        wt = np.abs(x) / np.abs(x).sum()
        base["_b"] = p @ wt
        return sum_matched(base, "_b")

    res = minimize(obj, np.full(len(MODELS), 1 / len(MODELS)), method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-6))
    weights = np.abs(res.x) / np.abs(res.x).sum()
    base["_b"] = p @ weights
    mult = base["target"].sum() / base["_b"].sum()
    cfg = dict(
        weights=dict(zip(MODELS, np.round(weights, 4).tolist(), strict=True)),
        iterations=iters,
        level_mult=round(float(mult), 5),
        val_sum_matched=round(float(res.fun), 5),
        val_score_calibrated=round(score(base["target"], base["_b"] * mult), 5),
        val_score_raw=round(score(base["target"], base["_b"]), 5),
    )
    print("[val]", json.dumps(cfg, ensure_ascii=False), flush=True)
    return cfg


# ---------------------------------------------------------------- этап 4
def ratio_level(panel, a: int, days: int = 7) -> float:
    """Оценка суммы на 11:00-14:30 по утру: утро(00:00-10:30) x среднее отношение за прошлые дни."""
    g = panel.y.sum(axis=0)
    n = a % STEPS_PER_DAY + 1
    ratios = [
        g[a + 1 - STEPS_PER_DAY * t : a + 1 + HORIZON - STEPS_PER_DAY * t].sum()
        / g[a + 1 - n - STEPS_PER_DAY * t : a + 1 - STEPS_PER_DAY * t].sum()
        for t in range(1, days + 1)
    ]
    return float(g[a + 1 - n : a + 1].sum() * np.mean(ratios))


def final(cfg: dict, level_mode: str, out_path: Path) -> None:
    t0 = time.time()
    panel = load_panel(ROOT / "Data" / "train_solo_track.parquet")
    fb = FeatureBuilder(panel, v3=True)
    last = panel.n_times - 1
    slots = range(TEST_SLOT - RADIUS, TEST_SLOT + RADIUS + 1)
    anchors = np.sort(np.concatenate([fb.valid_anchors(slot=s) for s in slots]))
    tr = fb.build(anchors)
    te = fb.build(np.array([last]), with_target=False)
    cols = feature_columns(tr)
    w = tr["scale"].to_numpy()
    print(f"[final] train rows={len(tr)} ({time.time() - t0:.0f}s)", flush=True)

    blend = np.zeros(len(te))
    for name, spec in MODELS.items():
        feats = select_features(cols, spec["fs"])
        rounds = int(cfg["iterations"][name] * ITER_MARGIN) + 1
        if spec["lib"] == "cb":
            from catboost import CatBoostRegressor, Pool

            model = CatBoostRegressor(
                loss_function=spec["loss"], iterations=rounds, learning_rate=LR, depth=8,
                border_count=64, random_seed=42, thread_count=-1, verbose=0,
            )
            model.fit(Pool(tr[feats], tr["target_norm"], weight=w, cat_features=["dow"]))
        else:
            params = dict(learning_rate=LR, bagging_freq=1, lambda_l2=1.0, max_bin=63, verbose=-1, seed=42, **lgb_cfg(spec))
            model = lgb.train(params, lgb.Dataset(tr[feats], tr["target_norm"], weight=w, categorical_feature=["dow"]), rounds)
        pred = np.clip(model.predict(te[feats]), 0, None) * te["scale"].to_numpy()
        blend += cfg["weights"][name] * pred
        print(f"[final] {name:10s} rounds={rounds} sum={pred.sum() / 1e9:.4f} млрд ({time.time() - t0:.0f}s)", flush=True)

    model_level = blend.sum() * cfg["level_mult"]
    if level_mode == "validation":
        target_sum = model_level
    elif level_mode == "ratio":
        target_sum = ratio_level(panel, last)
    else:  # avg
        target_sum = (model_level + ratio_level(panel, last)) / 2
    y_pred = blend * target_sum / blend.sum()
    print(f"[final] уровень={level_mode}: сумма {target_sum / 1e9:.4f} млрд, множитель {target_sum / blend.sum():.4f}")

    test = pd.read_parquet(ROOT / "Data" / "test_solo_track.parquet")
    test["timestamp"] = pd.to_datetime(test["timestamp"])
    fc = te[["route_id", "k"]].assign(
        timestamp=panel.times[last] + pd.to_timedelta(te["k"] * 30, unit="min"), y_pred=y_pred
    )
    sub = (
        test.merge(fc[["route_id", "timestamp", "y_pred"]], on=["route_id", "timestamp"], how="left")
        .sort_values("id")[["id", "y_pred"]]
        .reset_index(drop=True)
    )
    assert len(sub) == len(test) and sub["id"].is_unique
    assert sub["y_pred"].notna().all() and (sub["y_pred"] >= 0).all()
    sub.to_csv(out_path, index=False)
    print(f"[final] saved {out_path}: rows={len(sub)} sum={sub['y_pred'].sum() / 1e9:.4f} млрд ({time.time() - t0:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["validation", "ratio", "avg"], default="validation",
                    help="как оценить общую сумму прогноза (по умолчанию - множитель с валидации)")
    ap.add_argument("--reuse-validation", action="store_true", help="взять веса/итерации из runs/pipeline.json")
    ap.add_argument("--out", default=str(ROOT / "test.csv"))
    args = ap.parse_args()

    cfg_path = RUNS / "pipeline.json"
    if args.reuse_validation and cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    else:
        cfg = validate(args.level)
        RUNS.mkdir(exist_ok=True)
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    final(cfg, args.level, Path(args.out))


if __name__ == "__main__":
    main()
