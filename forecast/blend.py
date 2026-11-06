"""Подбор весов ансамбля по валидационным прогнозам runs/val_<name>.parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from forecast.experiment import sum_matched
from forecast.features import score

RUNS = Path(__file__).resolve().parents[1] / "runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--with-seasonal", action="store_true", help="добавить сезонный прогноз (среднее 7 дней)")
    args = ap.parse_args()

    frames = [pd.read_parquet(RUNS / f"val_{n}.parquet") for n in args.names]
    base = frames[0][["route_id", "anchor", "k", "target"]].copy()
    cols = []
    for n, f in zip(args.names, frames, strict=True):
        assert (f[["route_id", "anchor", "k"]].to_numpy() == base[["route_id", "anchor", "k"]].to_numpy()).all()
        base[n] = f["pred"].to_numpy()
        cols.append(n)
    if args.with_seasonal:
        va = pd.read_parquet(RUNS / f"feat_r{args.radius}_va.parquet", columns=["seas_7d", "scale"])
        base["seasonal"] = (va["seas_7d"] * va["scale"]).to_numpy()
        cols.append("seasonal")

    for c in cols:
        print(f"{c:>14}: raw={score(base['target'], base[c]):.4f}  sum_matched={sum_matched(base, c):.4f}")

    p = base[cols].to_numpy()

    def obj(w: np.ndarray) -> float:
        w = np.abs(w) / np.abs(w).sum()
        base["_b"] = p @ w
        return sum_matched(base, "_b")

    res = minimize(obj, np.full(len(cols), 1 / len(cols)), method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-6))
    w = np.abs(res.x) / np.abs(res.x).sum()
    base["_b"] = p @ w
    print("weights:", dict(zip(cols, np.round(w, 3), strict=True)))
    print(f"blend: raw={score(base['target'], base['_b']):.4f}  sum_matched={res.fun:.4f}")
    eq = p.mean(axis=1)
    base["_e"] = eq
    print(f"equal-weight blend sum_matched={sum_matched(base, '_e'):.4f}")


if __name__ == "__main__":
    main()
