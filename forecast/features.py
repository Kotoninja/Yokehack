"""Построение признаков для прогноза target_1h на 8 шагов (по 30 минут) вперёд.

Данные хранятся как матрицы [маршрут x время]: ряд у каждого маршрута полный,
с шагом 30 минут и общим началом, поэтому окна и лаги считаются срезами.

Строка обучающей выборки = (маршрут, якорь a, горизонт k):
  * якорь a - индекс последней известной точки;
  * прогнозируем Y[:, a + k], k = 1..8.
Все признаки используют только данные с индексами <= a.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STEPS_PER_DAY = 48
STEPS_PER_WEEK = 7 * STEPS_PER_DAY
HORIZON = 8
STATUS_COLS = [f"status_{i}" for i in range(1, 7)]
TARGET_COL = "target_1h"


@dataclass
class Panel:
    route_ids: np.ndarray  # (R,)
    times: pd.DatetimeIndex  # (T,)
    y: np.ndarray  # (R, T) float64
    s: np.ndarray  # (6, R, T) float64

    @property
    def n_times(self) -> int:
        return self.y.shape[1]


def load_panel(path: Path) -> Panel:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["route_id", "timestamp"])
    route_ids = np.sort(df["route_id"].unique())
    times = pd.DatetimeIndex(np.sort(df["timestamp"].unique()))
    n_r, n_t = len(route_ids), len(times)
    if len(df) != n_r * n_t:
        raise ValueError("ожидается полная сетка маршрут x время без пропусков")
    if times[0].hour != 0 or times[0].minute != 0:
        raise ValueError("ожидается, что ряд начинается в 00:00")
    y = df[TARGET_COL].to_numpy(np.float64).reshape(n_r, n_t)
    s = np.stack([df[c].to_numpy(np.float64).reshape(n_r, n_t) for c in STATUS_COLS])
    return Panel(route_ids, times, y, s)


class _Windows:
    """Быстрые средние по окнам через кумулятивные суммы."""

    def __init__(self, x: np.ndarray):
        pad = np.zeros(x.shape[:-1] + (1,))
        self.cs = np.concatenate([pad, np.cumsum(x, axis=-1)], axis=-1)
        self.cs2 = np.concatenate([pad, np.cumsum(x * x, axis=-1)], axis=-1)

    def mean(self, a: int, w: int) -> np.ndarray:
        """Среднее по индексам [a - w + 1, a]."""
        return (self.cs[..., a + 1] - self.cs[..., a + 1 - w]) / w

    def std(self, a: int, w: int) -> np.ndarray:
        m = self.mean(a, w)
        m2 = (self.cs2[..., a + 1] - self.cs2[..., a + 1 - w]) / w
        return np.sqrt(np.maximum(m2 - m * m, 0))


MIN_HISTORY = 4 * STEPS_PER_WEEK  # 28 дней истории до якоря


class FeatureBuilder:
    def __init__(self, panel: Panel, v2: bool = False):
        self.p = panel
        self.v2 = v2
        self.yw = _Windows(panel.y)
        self.sw = _Windows(panel.s)
        self.gy = panel.y.sum(axis=0)  # суммарный объём по всем маршрутам
        self.gw = _Windows(self.gy)

    def valid_anchors(self, slot: int | None = None, with_target: bool = True) -> np.ndarray:
        last = self.p.n_times - 1 - (HORIZON if with_target else 0)
        anchors = np.arange(MIN_HISTORY, last + 1)
        if slot is not None:
            anchors = anchors[anchors % STEPS_PER_DAY == slot]
        return anchors

    def build(self, anchors: np.ndarray, with_target: bool = True) -> pd.DataFrame:
        frames = [self._anchor_frame(int(a), with_target) for a in anchors]
        return pd.concat(frames, ignore_index=True)

    def _v2_base(self, a: int, scale: np.ndarray) -> dict[str, np.ndarray]:
        """Доп. признаки: темп сегодняшнего дня и динамика статусов."""
        y, d, w = self.p.y, STEPS_PER_DAY, STEPS_PER_WEEK
        slot_a = a % d
        n = slot_a + 1  # сколько точек сегодня уже известно (с 00:00 до a)
        out: dict[str, np.ndarray] = {}
        today = y[:, a - n + 1 : a + 1].sum(axis=1)
        prev = np.mean([y[:, a - n + 1 - d * t : a + 1 - d * t].sum(axis=1) for t in range(1, 8)], axis=0)
        week = y[:, a - n + 1 - w : a + 1 - w].sum(axis=1)
        out["today_vs_7d"] = today / (prev + scale)
        out["today_vs_w1"] = today / (week + scale)
        g = self.gy
        g_today = g[a - n + 1 : a + 1].sum()
        g_prev = np.mean([g[a - n + 1 - d * t : a + 1 - d * t].sum() for t in range(1, 8)])
        out["g_today_vs_7d"] = np.full(y.shape[0], g_today / g_prev)
        # доля остатка дня, которая обычно приходится на 11:00-14:30, считается в seas-признаках
        for i in range(6):
            s = self.p.s[i]
            m7 = self.sw.mean(a, w)[i] + 1.0
            name = f"s{i + 1}"
            out[f"{name}_d1h"] = (s[:, a] - s[:, a - 2]) / m7
            out[f"{name}_d1d"] = (s[:, a] - s[:, a - d]) / m7
            out[f"{name}_today"] = s[:, a - n + 1 : a + 1].mean(axis=1) / m7
        s = self.p.s
        out["s4_over_s5"] = s[3, :, a] / (s[4, :, a] + 1)
        out["s123_4h_to_y"] = (self.sw.mean(a, 8)[:3].sum(axis=0)) / scale
        out["route_cat"] = self.p.route_ids
        return out

    def _anchor_frame(self, a: int, with_target: bool) -> pd.DataFrame:
        p, y, yw, sw = self.p, self.p.y, self.yw, self.sw
        n_r = y.shape[0]
        d, w = STEPS_PER_DAY, STEPS_PER_WEEK

        scale = yw.mean(a, w) + 1.0  # уровень маршрута за 7 дней
        base: dict[str, np.ndarray] = {
            "route_id": p.route_ids,
            "anchor": np.full(n_r, a),
            "scale": scale,
            "log_scale": np.log(scale),
            "lvl_1h": yw.mean(a, 2) / scale,
            "lvl_2h": yw.mean(a, 4) / scale,
            "lvl_4h": yw.mean(a, 8) / scale,
            "lvl_12h": yw.mean(a, 24) / scale,
            "lvl_1d": yw.mean(a, d) / scale,
            "lvl_3d": yw.mean(a, 3 * d) / scale,
            "lvl_14d": yw.mean(a, 2 * w) / scale,
            "lvl_28d": yw.mean(a, 4 * w) / scale,
            "std_7d": yw.std(a, w) / scale,
            "std_1d": yw.std(a, d) / scale,
            "zero_frac_7d": (y[:, a - w + 1 : a + 1] == 0).mean(axis=1),
        }
        for lag in range(4):
            base[f"y_lag{lag}"] = y[:, a - lag] / scale

        g7 = self.gw.mean(a, w)
        base["g_lvl_4h"] = np.full(n_r, self.gw.mean(a, 8) / g7)
        base["g_lvl_1d"] = np.full(n_r, self.gw.mean(a, d) / g7)
        base["g_lvl_28d"] = np.full(n_r, self.gw.mean(a, 4 * w) / g7)

        for i in range(6):
            m7 = sw.mean(a, w)[i] + 1.0
            name = f"s{i + 1}"
            base[f"{name}_log7d"] = np.log(m7)
            base[f"{name}_to_y"] = m7 / scale
            base[f"{name}_last"] = p.s[i, :, a] / m7
            base[f"{name}_1h"] = sw.mean(a, 2)[i] / m7
            base[f"{name}_4h"] = sw.mean(a, 8)[i] / m7
            base[f"{name}_1d"] = sw.mean(a, d)[i] / m7

        if self.v2:
            base.update(self._v2_base(a, scale))
        base_df = pd.DataFrame(base)
        frames = []
        for k in range(1, HORIZON + 1):
            j = a + k
            slot = j % d
            day_idx = j // d
            f = base_df.copy()
            f["k"] = k
            f["slot"] = slot
            # данные начинаются в понедельник 2025-07-28
            f["dow"] = day_idx % 7
            f["y_d1"] = y[:, j - d] / scale
            f["y_d2"] = y[:, j - 2 * d] / scale
            f["y_w1"] = y[:, j - w] / scale
            f["y_w2"] = y[:, j - 2 * w] / scale
            f["y_d1_nb"] = y[:, j - d - 1 : j - d + 2].mean(axis=1) / scale
            f["seas_7d"] = np.mean([y[:, j - d * t] for t in range(1, 8)], axis=0) / scale
            f["seas_4w"] = np.mean([y[:, j - w * t] for t in range(1, 5)], axis=0) / scale
            f["seas_28d"] = np.mean([y[:, j - d * t] for t in range(1, 29)], axis=0) / scale
            # профиль маршрута: доля слота относительно среднего дня
            f["slot_prof_28d"] = f["seas_28d"] / base_df["lvl_28d"].clip(lower=1e-6)
            f["g_seas_7d"] = np.mean([self.gy[j - d * t] for t in range(1, 8)]) / g7
            if self.v2:
                # «сегодняшний уровень x вчерашняя форма»: текущие 2ч, перенесённые
                # по отношению сезонного профиля в целевом слоте к профилю в последние 2ч
                prof_now = np.mean(
                    [y[:, a - 3 - d * t : a + 1 - d * t].mean(axis=1) for t in range(1, 8)], axis=0
                )
                f["shape_fc"] = base_df["lvl_2h"] * f["seas_7d"] / (prof_now / scale + 1e-3)
            if with_target:
                f["target"] = y[:, j]
                f["target_norm"] = y[:, j] / scale
            f["t_idx"] = j
            frames.append(f)
        return pd.concat(frames, ignore_index=True)


NON_FEATURES = {"route_id", "anchor", "scale", "target", "target_norm", "t_idx"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


def score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WAPE + |Relative Bias|."""
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    s = y_true.sum()
    return np.abs(y_pred - y_true).sum() / s + abs(y_pred.sum() / s - 1)
