"""
Test the exported xG model on StatsBomb Open Data Champions League.

Important:
- StatsBomb Open Data does NOT include Champions League 2023/24 or 2024.
- In open data, UCL coverage is VERY LIMITED (often just 1 match per season,
  typically the final). For a more stable estimate, this script defaults to
  testing on ALL open UCL matches across all available seasons.

This script:
1) Loads exported model + preprocess + metadata (joblib/json).
2) Downloads open UCL matches + shot events (cached to data/).
3) Rebuilds the same engineered features used in training (incl. freeze-frame).
4) Evaluates: AUC-ROC, PR-AUC, LogLoss, Brier, and compares to StatsBomb xG.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from statsbombpy import sb


GOAL_X = 120.0
GOAL_Y = 40.0
LEFT_POST_Y = 36.0
RIGHT_POST_Y = 44.0


def shot_distance_and_angle(x: float, y: float) -> Tuple[float, float]:
    dx = GOAL_X - x
    dy = GOAL_Y - y
    dist = math.sqrt(dx * dx + dy * dy)
    a1 = math.atan2(RIGHT_POST_Y - y, GOAL_X - x)
    a2 = math.atan2(LEFT_POST_Y - y, GOAL_X - x)
    ang = abs(a1 - a2)
    return dist, ang


def lane_bounds_at_x(shooter_x: float, shooter_y: float, x: float) -> Optional[Tuple[float, float]]:
    if x <= shooter_x:
        return None
    if GOAL_X == shooter_x:
        return None
    t = (x - shooter_x) / (GOAL_X - shooter_x)
    y_left = shooter_y + t * (LEFT_POST_Y - shooter_y)
    y_right = shooter_y + t * (RIGHT_POST_Y - shooter_y)
    return (min(y_left, y_right), max(y_left, y_right))


def freeze_frame_features(
    shooter_x: float, shooter_y: float, freeze_frame: Any
) -> Dict[str, Any]:
    if not isinstance(freeze_frame, list) or len(freeze_frame) == 0:
        return {
            "gk_distance": np.nan,
            "nearest_defender_distance": np.nan,
            "defenders_within_1": 0,
            "defenders_within_2": 0,
            "defenders_within_3": 0,
            "defenders_in_lane": 0,
        }

    defenders = [
        p
        for p in freeze_frame
        if p.get("teammate") is False and isinstance(p.get("location"), list)
    ]

    gk = None
    for p in defenders:
        if (p.get("position") or {}).get("name") == "Goalkeeper":
            gk = p
            break

    def dist(loc: List[float]) -> float:
        return math.sqrt((loc[0] - shooter_x) ** 2 + (loc[1] - shooter_y) ** 2)

    gk_dist = dist(gk["location"]) if gk is not None else np.nan

    defenders_wo_gk = [p for p in defenders if p is not gk]
    dists = [dist(p["location"]) for p in defenders_wo_gk]
    nearest_def = float(min(dists)) if len(dists) else np.nan

    within_1 = int(sum(d <= 1.0 for d in dists))
    within_2 = int(sum(d <= 2.0 for d in dists))
    within_3 = int(sum(d <= 3.0 for d in dists))

    in_lane = 0
    for p in defenders_wo_gk:
        x, y = p["location"]
        if x <= shooter_x or x > GOAL_X:
            continue
        bounds = lane_bounds_at_x(shooter_x, shooter_y, x)
        if bounds is None:
            continue
        y_low, y_high = bounds
        if y_low <= y <= y_high:
            in_lane += 1

    return {
        "gk_distance": gk_dist,
        "nearest_defender_distance": nearest_def,
        "defenders_within_1": within_1,
        "defenders_within_2": within_2,
        "defenders_within_3": within_3,
        "defenders_in_lane": int(in_lane),
    }


def build_ucl_open_shots_season(cache_path: Path, season_id: int) -> pd.DataFrame:
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        return df

    competition_id = 16
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    match_ids = matches["match_id"].astype(int).tolist()
    print(f"Fetching Champions League season_id={season_id}: {len(match_ids)} matches")

    rows: List[Dict[str, Any]] = []

    for match_id in match_ids:
        ev = sb.events(match_id=match_id)
        shots = ev[ev["type"] == "Shot"].copy()
        if len(shots) == 0:
            continue

        for _, r in shots.iterrows():
            loc = r.get("location")
            if not isinstance(loc, list) or len(loc) < 2:
                continue
            x, y = float(loc[0]), float(loc[1])
            dist, ang = shot_distance_and_angle(x, y)

            ff_feats = freeze_frame_features(x, y, r.get("shot_freeze_frame"))

            shot_type = r.get("shot_type")
            outcome = r.get("shot_outcome")
            goal = int(outcome == "Goal")

            statsbomb_xg = r.get("shot_statsbomb_xg")

            rows.append(
                {
                    "match_id": int(r.get("match_id")) if pd.notna(r.get("match_id")) else int(match_id),
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "team": r.get("team"),
                    "player": r.get("player"),
                    "x": x,
                    "y": y,
                    "distance_to_goal": dist,
                    "angle_to_goal": ang,
                    "under_pressure": int(bool(r.get("under_pressure"))) if pd.notna(r.get("under_pressure")) else 0,
                    "play_pattern": r.get("play_pattern"),
                    "shot_type": shot_type,
                    "body_part": r.get("shot_body_part"),
                    "shot_technique": r.get("shot_technique"),
                    "shot_first_time": int(bool(r.get("shot_first_time"))) if pd.notna(r.get("shot_first_time")) else 0,
                    "shot_open_goal": int(bool(r.get("shot_open_goal"))) if pd.notna(r.get("shot_open_goal")) else 0,
                    "shot_follows_dribble": int(bool(r.get("shot_follows_dribble"))) if pd.notna(r.get("shot_follows_dribble")) else 0,
                    "shot_deflected": int(bool(r.get("shot_deflected"))) if pd.notna(r.get("shot_deflected")) else 0,
                    "is_penalty": int(shot_type == "Penalty"),
                    "is_free_kick_shot": int(shot_type == "Free Kick"),
                    "statsbomb_xg": float(statsbomb_xg) if pd.notna(statsbomb_xg) else np.nan,
                    "goal": goal,
                    **ff_feats,
                }
            )

    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def build_ucl_open_shots_all(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)

    competition_id = 16
    comps = sb.competitions()
    cl = comps[comps["competition_id"] == competition_id][["season_id", "season_name"]].drop_duplicates()
    cl = cl.sort_values("season_name")

    all_rows: List[Dict[str, Any]] = []
    total_matches = 0

    for _, row in cl.iterrows():
        sid = int(row["season_id"])
        sname = str(row["season_name"])
        matches = sb.matches(competition_id=competition_id, season_id=sid)
        match_ids = matches["match_id"].astype(int).tolist()
        total_matches += len(match_ids)
        print(f"Fetching UCL {sname} (season_id={sid}): {len(match_ids)} matches")

        for match_id in match_ids:
            ev = sb.events(match_id=match_id)
            shots = ev[ev["type"] == "Shot"].copy()
            if len(shots) == 0:
                continue

            for _, r in shots.iterrows():
                loc = r.get("location")
                if not isinstance(loc, list) or len(loc) < 2:
                    continue
                x, y = float(loc[0]), float(loc[1])
                dist, ang = shot_distance_and_angle(x, y)

                ff_feats = freeze_frame_features(x, y, r.get("shot_freeze_frame"))

                shot_type = r.get("shot_type")
                outcome = r.get("shot_outcome")
                goal = int(outcome == "Goal")
                statsbomb_xg = r.get("shot_statsbomb_xg")

                all_rows.append(
                    {
                        "match_id": int(r.get("match_id")) if pd.notna(r.get("match_id")) else int(match_id),
                        "competition_id": competition_id,
                        "season_id": sid,
                        "season_name": sname,
                        "team": r.get("team"),
                        "player": r.get("player"),
                        "x": x,
                        "y": y,
                        "distance_to_goal": dist,
                        "angle_to_goal": ang,
                        "under_pressure": int(bool(r.get("under_pressure"))) if pd.notna(r.get("under_pressure")) else 0,
                        "play_pattern": r.get("play_pattern"),
                        "shot_type": shot_type,
                        "body_part": r.get("shot_body_part"),
                        "shot_technique": r.get("shot_technique"),
                        "shot_first_time": int(bool(r.get("shot_first_time"))) if pd.notna(r.get("shot_first_time")) else 0,
                        "shot_open_goal": int(bool(r.get("shot_open_goal"))) if pd.notna(r.get("shot_open_goal")) else 0,
                        "shot_follows_dribble": int(bool(r.get("shot_follows_dribble"))) if pd.notna(r.get("shot_follows_dribble")) else 0,
                        "shot_deflected": int(bool(r.get("shot_deflected"))) if pd.notna(r.get("shot_deflected")) else 0,
                        "is_penalty": int(shot_type == "Penalty"),
                        "is_free_kick_shot": int(shot_type == "Free Kick"),
                        "statsbomb_xg": float(statsbomb_xg) if pd.notna(statsbomb_xg) else np.nan,
                        "goal": goal,
                        **ff_feats,
                    }
                )

    df = pd.DataFrame(all_rows)
    print(f"Total open UCL matches fetched: {total_matches}")
    cache_path.parent.mkdir(exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models"
    data_dir = base_dir / "data"

    meta_path = models_dir / "xg_metadata_20260204_234629.json"
    model_path = models_dir / "xg_model_xgb_20260204_234629.joblib"
    preprocess_path = models_dir / "xg_preprocess_20260204_234629.joblib"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_cols = meta["feature_cols"]

    model = joblib.load(model_path)
    preprocess = joblib.load(preprocess_path)

    # Build CL shots (default: ALL open UCL matches, across all seasons)
    cl_cache = data_dir / "ucl_open_all_seasons_shots.csv"
    df = build_ucl_open_shots_all(cl_cache)
    print(f"UCL(open) shots: {len(df):,} | goals: {int(df['goal'].sum()):,} | rate: {df['goal'].mean():.2%}")

    # Prepare X/y
    X_df = df[feature_cols].copy()
    y = df["goal"].astype(int).values

    X = preprocess.transform(X_df)
    proba = model.predict_proba(X)[:, 1]

    # Metrics
    out = {
        "AUC-ROC": roc_auc_score(y, proba),
        "PR-AUC": average_precision_score(y, proba),
        "LogLoss": log_loss(y, proba),
        "Brier": brier_score_loss(y, proba),
    }

    print("\n✅ Exported-model performance on UCL (open data, all seasons)")
    for k, v in out.items():
        print(f"{k:8s}: {v:.4f}")

    # Baseline comparison: StatsBomb xG (if available)
    if "statsbomb_xg" in df.columns:
        base = df["statsbomb_xg"].astype(float)
        ok = base.notna().values
        if ok.sum() > 0:
            base_out = {
                "AUC-ROC": roc_auc_score(y[ok], base.values[ok]),
                "PR-AUC": average_precision_score(y[ok], base.values[ok]),
                "LogLoss": log_loss(y[ok], base.values[ok]),
                "Brier": brier_score_loss(y[ok], base.values[ok]),
            }
            print("\n📏 Baseline on same shots: StatsBomb xG")
            for k, v in base_out.items():
                print(f"{k:8s}: {v:.4f}")

    # Save predictions for inspection
    pred_path = data_dir / "ucl_open_all_seasons_predictions.csv"
    df_out = df.copy()
    df_out["pred_xg"] = proba
    df_out.to_csv(pred_path, index=False)
    print(f"\nSaved predictions: {pred_path}")


if __name__ == "__main__":
    main()

# ✅ Exported-model performance on UCL (open data, all seasons)
# AUC-ROC : 0.8686
# PR-AUC  : 0.5450
# LogLoss : 0.2575
# Brier   : 0.0752

# 📏 Baseline on same shots: StatsBomb xG
# AUC-ROC : 0.8733
# PR-AUC  : 0.5588
# LogLoss : 0.2545
# Brier   : 0.0734