import os
import json
import glob
import csv
import argparse
import numpy as np
from collections import defaultdict


def create_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def mean_std(vals):
    vals = np.array(vals, dtype=float)
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


METRICS_ORDER = ["auc", "acc", "precision", "recall", "f1", "auprc"]  # auprc 마지막 고정


def _first_level_dir(seed_dir: str, file_path: str) -> str:
    """
    seed_dir/CONFIG/.../file.json  -> CONFIG
    """
    rel = os.path.relpath(os.path.dirname(file_path), seed_dir)
    first = rel.split(os.sep)[0] if rel else ""
    return first


def _extract_metrics_from_block(d: dict):
    """
    키가 뭐든 간에 마지막이 _auc, _acc, ... 로 끝나면 metric으로 인식.
    예: Ours_best_full_auc, cat_best_few_auc 등 전부 처리됨.
    """
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if not isinstance(k, str):
            continue
        for m in METRICS_ORDER:
            if k.endswith(f"_{m}"):
                out[m] = v
    return out


def _extract_one_json(data: dict):
    """
    다양한 저장 포맷을 최대한 수용해서
    - few_shot
    - batch_size
    - few_metrics (dict metric->value)
    - full_metrics (dict metric->value)
    를 뽑아냄.
    """
    hp = data.get("hyperparameters", data.get("args", {}))
    few_shot = hp.get("few_shot", data.get("few_shot", None))
    batch_size = hp.get("batch_size", data.get("batch_size", None))

    res = data.get("results", {})

    few_metrics = {}
    full_metrics = {}

    # (A) 너가 쓰던 DL 스타일: results = {"Ours":{...}, "Ours_few":{...}}
    if isinstance(res, dict):
        if "Ours_few" in res:
            few_metrics = _extract_metrics_from_block(res.get("Ours_few", {}))
        if "Ours" in res:
            full_metrics = _extract_metrics_from_block(res.get("Ours", {}))

    # (B) 혹시 Best_results 구조인 경우도 지원
    # results = {"Best_results": {"few": {...}, "full": {...}}}
    if (not few_metrics or not full_metrics) and isinstance(res, dict) and "Best_results" in res:
        br = res.get("Best_results", {})
        few_block = br.get("few", {})
        full_block = br.get("full", {})

        # few/full 내부가 dict이고, 그 안에 Ours 같은 키가 있든 없든 metric suffix로 뽑음
        if not few_metrics:
            if isinstance(few_block, dict):
                # Ours 키가 있으면 그걸 우선
                if "Ours_few" in few_block:
                    few_metrics = _extract_metrics_from_block(few_block.get("Ours_few", {}))
                elif "Ours" in few_block:
                    few_metrics = _extract_metrics_from_block(few_block.get("Ours", {}))
                else:
                    few_metrics = _extract_metrics_from_block(few_block)

        if not full_metrics:
            if isinstance(full_block, dict):
                if "Ours" in full_block:
                    full_metrics = _extract_metrics_from_block(full_block.get("Ours", {}))
                else:
                    full_metrics = _extract_metrics_from_block(full_block)

    # 최소 조건
    if few_shot is None:
        return None
    if batch_size is None:
        # batch_size가 json에 없을 수도 있으니 None 허용 -> 열 정렬에만 영향
        batch_size = "NA"

    # few/full 둘 중 하나라도 있으면 유효
    if not few_metrics and not full_metrics:
        return None

    return str(few_shot), str(batch_size), few_metrics, full_metrics


def summarize_seed_ignore_configs(base_dir: str):
    # 1) seed 폴더 찾기
    seed_dirs = sorted(glob.glob(os.path.join(base_dir, "args_seed:*")))
    seed_dirs = [d for d in seed_dirs if os.path.isdir(d)]
    if not seed_dirs:
        raise RuntimeError(f"No args_seed:* directories found under: {base_dir}")

    # 2) seed별로: config 평균(=seed 하나당 값) 만들기 위한 중간 저장소
    # seed -> config -> few_shot -> metric -> value
    seed_config_few = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    # seed -> config -> metric -> value  (full은 config당 1번만 쓰게)
    seed_config_full = defaultdict(lambda: defaultdict(dict))

    # few-shot 키 모으기 (전체 seed 통합)
    all_few_shots = set()

    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)

        # json 찾기: seed/config/.../*.json
        json_files = glob.glob(os.path.join(seed_dir, "**/*.json"), recursive=True)
        if not json_files:
            continue

        # 같은 (config, few_shot)에 json이 여러 개면 최신(mtime)만 쓰도록
        best_json_by_run = {}  # (config, few_shot) -> (mtime, path)
        best_json_by_config_for_full = {}  # config -> (mtime, path)

        for jf in json_files:
            try:
                with open(jf, "r") as f:
                    data = json.load(f)
            except Exception:
                continue

            extracted = _extract_one_json(data)
            if extracted is None:
                continue

            few_shot, batch_size, few_metrics, full_metrics = extracted
            all_few_shots.add(few_shot)

            cfg = _first_level_dir(seed_dir, jf)
            mtime = os.path.getmtime(jf)

            run_key = (cfg, few_shot)
            # 최신 json 유지
            if run_key not in best_json_by_run or mtime > best_json_by_run[run_key][0]:
                best_json_by_run[run_key] = (mtime, jf)

            # full은 config 기준으로 1개만(중복 방지) -> 최신 json 유지
            if full_metrics:
                if cfg not in best_json_by_config_for_full or mtime > best_json_by_config_for_full[cfg][0]:
                    best_json_by_config_for_full[cfg] = (mtime, jf)

        # 이제 선택된 json들만 다시 읽어서 seed_config_* 채우기
        for (cfg, few_shot), (_, jf) in best_json_by_run.items():
            with open(jf, "r") as f:
                data = json.load(f)
            extracted = _extract_one_json(data)
            if extracted is None:
                continue
            few_shot, batch_size, few_metrics, full_metrics = extracted

            # few: config별 단일 값 저장
            for m, v in few_metrics.items():
                seed_config_few[seed_name][cfg][few_shot][m] = v

        for cfg, (_, jf) in best_json_by_config_for_full.items():
            with open(jf, "r") as f:
                data = json.load(f)
            extracted = _extract_one_json(data)
            if extracted is None:
                continue
            few_shot, batch_size, few_metrics, full_metrics = extracted

            for m, v in full_metrics.items():
                seed_config_full[seed_name][cfg][m] = v

    # 3) seed 하나당 값 만들기: (seed 내부 config 평균)
    # seed -> few_shot -> metric -> seed_mean_value
    seed_level_few = defaultdict(lambda: defaultdict(dict))
    # seed -> metric -> seed_mean_value
    seed_level_full = defaultdict(dict)

    for seed_name, cfg_dict in seed_config_few.items():
        # few-shot
        for few_shot in all_few_shots:
            for m in METRICS_ORDER:
                vals = []
                for cfg, fs_dict in cfg_dict.items():
                    if few_shot in fs_dict and m in fs_dict[few_shot]:
                        vals.append(fs_dict[few_shot][m])
                if vals:
                    seed_level_few[seed_name][few_shot][m] = float(np.mean(vals))

        # full
        full_vals_by_metric = defaultdict(list)
        for cfg, met in seed_config_full[seed_name].items():
            for m in METRICS_ORDER:
                if m in met:
                    full_vals_by_metric[m].append(met[m])

        for m in METRICS_ORDER:
            if full_vals_by_metric[m]:
                seed_level_full[seed_name][m] = float(np.mean(full_vals_by_metric[m]))

    # 4) 최종: seed들 사이 mean±std
    # few_shot -> metric -> list(seed_mean)
    agg_few = defaultdict(lambda: defaultdict(list))
    agg_full = defaultdict(list)

    for seed_name in seed_level_few.keys():
        for few_shot, met in seed_level_few[seed_name].items():
            for m in METRICS_ORDER:
                if m in met:
                    agg_few[few_shot][m].append(met[m])

        for m in METRICS_ORDER:
            if m in seed_level_full[seed_name]:
                agg_full[m].append(seed_level_full[seed_name][m])

    few_shots_sorted = sorted(list(all_few_shots), key=lambda x: int(x) if str(x).isdigit() else str(x))

    # 5) 파일 저장: 딱 하나
    out_dir = os.path.join(base_dir, "summary")
    create_directory(out_dir)
    out_path = os.path.join(out_dir, "all_configs_combined.tsv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        # 행=metric, 열=few-shot + full
        for m in METRICS_ORDER:
            row = []
            for fs in few_shots_sorted:
                vals = agg_few[fs][m]
                if vals:
                    mu, sd = mean_std(vals)
                    row.append(f"{mu:.4f}({sd:.4f})")
                else:
                    row.append("")
            # full(맨 마지막)
            if agg_full[m]:
                mu, sd = mean_std(agg_full[m])
                row.append(f"{mu:.4f}({sd:.4f})")
            else:
                row.append("")
            writer.writerow(row)

    print(f"[OK] Saved: {out_path}")
    print(f"      columns: few_shot={few_shots_sorted} + [full]")
    print(f"      rows: {METRICS_ORDER}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    args = parser.parse_args()

    summarize_seed_ignore_configs(args.base_dir)


if __name__ == "__main__":
    main()
