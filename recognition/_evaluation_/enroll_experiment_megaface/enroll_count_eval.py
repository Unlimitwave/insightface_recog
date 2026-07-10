"""FaceScrub 多注册张数对比实验（1:N 识别 + 1:1 验证）。

实验问题：每人注册 1/3/5/10 张照片时，识别/验证指标如何变化？

数据划分（两种评测共用）：
  - 按文件名排序，每人前 k 张 → 注册（gallery）
  - 剩余照片 → 探针（probe），模拟现场抓拍
  - 同一人多张注册照比对时取 Max 相似度

1:N 指标：Rank-1 / Rank-5 / mAP / TPIR@FPIR / FNIR / FPIR
1:1 指标：TAR@FAR / FRR@FAR / EER / 最佳阈值 Accuracy

前置：先运行 recognition/_evaluation_/megaface/run_buffalo_l.sh 提取特征。
"""

from __future__ import print_function

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

from enroll_metrics import (
    FPIR_TARGETS_1N,
    FAR_TARGETS_11,
    evaluate_identification_1n,
    evaluate_verification_11,
    far_label,
)

# 引入 MegaFace 特征加载工具（读取预提取 .bin 特征）
_MEGAface_DIR = os.path.join(os.path.dirname(__file__), "..", "megaface")
sys.path.insert(0, os.path.abspath(_MEGAface_DIR))
from megaface_metrics import _feature_path, load_feature_bin  # noqa: E402


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def load_noise_names(noise_path):
    """读取 FaceScrub 噪声标注，过滤错误标签图片。"""
    if not noise_path or not os.path.isfile(noise_path):
        return set()
    names = set()
    with open(noise_path, "r") as fp:
        for line in fp:
            line = line.strip()
            if line and not line.startswith("#"):
                names.add(line)
    return names


def collect_facescrub_by_identity(facescrub_root, noise_names):
    """扫描 facescrub_images/，按身份收集图片路径（排序、去噪）。"""
    identities = {}
    for identity in sorted(os.listdir(facescrub_root)):
        identity_dir = os.path.join(facescrub_root, identity)
        if not os.path.isdir(identity_dir):
            continue
        images = []
        for name in sorted(os.listdir(identity_dir)):
            if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            if name in noise_names:
                continue
            images.append("%s/%s" % (identity, name))
        if images:
            identities[identity] = images
    return identities


def normalize_rows(feats):
    """L2 归一化，使点积 = 余弦相似度。"""
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return feats / norms


def load_facescrub_features(feature_root, identities, algo):
    """加载 FaceScrub 全部预提取特征，返回 {相对路径: 512维向量}。"""
    cache = {}
    missing = 0
    for identity, paths in identities.items():
        for rel_path in paths:
            feat_path = _feature_path(feature_root, rel_path, algo, "facescrub")
            if not os.path.isfile(feat_path):
                missing += 1
                continue
            feat = load_feature_bin(feat_path)
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            cache[rel_path] = feat
    if missing:
        print("  warning: missing %d facescrub features" % missing)
    return cache


def distractor_cache_path(feature_root, gallery_size):
    """MegaFace distractor npz 缓存路径。"""
    return os.path.join(feature_root, "distractor_cache_%d.npz" % gallery_size)


def load_distractor_features(feature_root, megaface_lst, algo, gallery_size, force_reload=False):
    """加载 MegaFace distractor 特征矩阵 (N, 512)，优先读 npz 缓存。"""
    cache_path = distractor_cache_path(feature_root, gallery_size)
    if not force_reload and os.path.isfile(cache_path):
        print("  loading distractor cache: %s" % cache_path)
        t0 = time.time()
        data = np.load(cache_path)
        feats = data["feats"].astype(np.float32)
        print("  loaded %d cached distractors in %.1fs" % (feats.shape[0], time.time() - t0))
        return feats

    paths = []
    with open(megaface_lst, "r") as fp:
        for line in fp:
            line = line.strip()
            if line:
                paths.append(line)
    paths = paths[:gallery_size]

    feats = []
    missing = 0
    for idx, rel_path in enumerate(paths):
        feat_path = _feature_path(feature_root, rel_path, algo, "megaface")
        if not os.path.isfile(feat_path):
            missing += 1
            continue
        feat = load_feature_bin(feat_path)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        feats.append(feat)
        if (idx + 1) % 100000 == 0:
            print("  loaded distractors: %d/%d" % (len(feats), len(paths)))
    if missing:
        print("  warning: missing %d distractor features" % missing)
    if not feats:
        raise RuntimeError("No distractor features loaded.")
    feats = normalize_rows(np.stack(feats, axis=0).astype(np.float32))
    np.savez_compressed(cache_path, feats=feats)
    print("  saved distractor cache: %s" % cache_path)
    return feats


# ---------------------------------------------------------------------------
# 核心评测：数据划分 + 相似度矩阵 + 指标计算
# ---------------------------------------------------------------------------


def build_enroll_probe_split(enroll_count, identities, feat_cache):
    """按注册张数 k 划分 enroll / probe，并构建特征矩阵。

    划分规则（以 k=3 为例）：
      Adam_Brody/001~003.jpg → 注册（3 个向量，绑定同一 identity）
      Adam_Brody/004~030.jpg → 探针（27 张，每张模拟一次现场抓拍）

    返回：
      enroll_matrix:     (总注册图数, 512)
      probe_matrix:      (总探针数, 512)
      enroll_identities: 每条注册图对应的 identity 名
      probe_identities:  每条探针对应的 identity 名
      skipped:           因照片不足而跳过的身份数
    """
    enroll_rows = []
    enroll_identities = []
    probe_rows = []
    probe_identities = []
    skipped = 0

    for identity, paths in sorted(identities.items()):
        available = [p for p in paths if p in feat_cache]
        # 至少需要 k 张注册 + 1 张探针，否则该身份跳过
        if len(available) <= enroll_count:
            skipped += 1
            continue
        for rel_path in available[:enroll_count]:
            enroll_rows.append(feat_cache[rel_path])
            enroll_identities.append(identity)
        for rel_path in available[enroll_count:]:
            probe_rows.append(feat_cache[rel_path])
            probe_identities.append(identity)

    if not enroll_rows or not probe_rows:
        raise RuntimeError("No valid enroll/probe split for k=%d" % enroll_count)

    enroll_matrix = normalize_rows(np.stack(enroll_rows, axis=0).astype(np.float32))
    probe_matrix = normalize_rows(np.stack(probe_rows, axis=0).astype(np.float32))
    return (
        enroll_matrix,
        probe_matrix,
        enroll_identities,
        probe_identities,
        skipped,
    )


def compute_identity_scores(enroll_matrix, probe_matrix, enroll_identities):
    """计算每个探针对每个身份的 Max 相似度。

    Step 1. probe @ enroll.T → (num_probes, num_enroll_templates) 逐模板相似度
    Step 2. 同一 identity 的多张注册照取 max → (num_probes, num_identities)

    例：k=3 时 Adam 有 3 行注册模板，probe 与这 3 行算分后取 max 作为 Adam 的得分。
    """
    enroll_index_by_identity = defaultdict(list)
    for idx, identity in enumerate(enroll_identities):
        enroll_index_by_identity[identity].append(idx)

    identity_list = sorted(enroll_index_by_identity.keys())
    identity_to_col = {name: idx for idx, name in enumerate(identity_list)}

    enroll_scores = probe_matrix.dot(enroll_matrix.T)
    num_probes = probe_matrix.shape[0]
    num_identities = len(identity_list)
    identity_scores = np.full((num_probes, num_identities), -1.0, dtype=np.float32)

    for identity, indices in enroll_index_by_identity.items():
        col = identity_to_col[identity]
        identity_scores[:, col] = enroll_scores[:, indices].max(axis=1)

    return identity_scores, identity_to_col, enroll_index_by_identity


def evaluate_enroll_count(enroll_count, identities, feat_cache, distractor_feats):
    """对单个注册张数 k 执行完整 1:N + 1:1 评测。

    流程概览：
      [1] 划分 enroll / probe
      [2] 批量矩阵乘法算相似度
      [3] Max 融合 → identity_scores (探针 × 身份)
      [4] 调用 enroll_metrics 计算全部标准化指标
    """
    # --- [1] 数据划分 ---
    (
        enroll_matrix,
        probe_matrix,
        enroll_identities,
        probe_identities,
        skipped,
    ) = build_enroll_probe_split(enroll_count, identities, feat_cache)

    # --- [2] 相似度矩阵 ---
    identity_scores, identity_to_col, enroll_index_by_identity = compute_identity_scores(
        enroll_matrix, probe_matrix, enroll_identities
    )
    probe_cols = np.array([identity_to_col[p] for p in probe_identities])

    # 探针 vs 全部 distractor：(num_probes, num_distractors)
    distractor_scores = probe_matrix.dot(distractor_feats.T)

    # --- [3] 1:N 指标 ---
    metrics_1n = evaluate_identification_1n(
        identity_scores, probe_cols, distractor_scores, FPIR_TARGETS_1N
    )

    # --- [4] 1:1 指标（复用 identity_scores，不含 distractor）---
    metrics_11 = evaluate_verification_11(
        identity_scores, probe_cols, FAR_TARGETS_11
    )

    return {
        "enroll_count": enroll_count,
        "identities": len(enroll_index_by_identity),
        "skipped_identities": skipped,
        "enroll_images": enroll_matrix.shape[0],
        "probe_images": probe_matrix.shape[0],
        "identification_1n": metrics_1n,
        "verification_11": metrics_11,
        # 顶层保留常用字段，便于脚本/表格直接读取
        "rank1": metrics_1n["rank1"],
        "rank5": metrics_1n["rank5"],
        "map": metrics_1n["map"],
        "verification_11_summary": {
            "positive_pairs": metrics_11["positive_pairs"],
            "negative_pairs": metrics_11["negative_pairs"],
            "tar_at_far": metrics_11["tar_at_far"],
            "frr_at_far": metrics_11["frr_at_far"],
            "eer": metrics_11["eer"],
        },
    }


# ---------------------------------------------------------------------------
# 结果展示
# ---------------------------------------------------------------------------


def _fmt_pct(val):
    return "%8.2f%%" % val if val is not None else "     N/A"


def print_summary_table(results, gallery_size):
    """打印 1:N / 1:1 汇总表。"""
    print("")
    print("=" * 108)
    print("1:N 识别 (distractors=%d) — Rank / mAP / TPIR@FPIR / FNIR(漏检) / FPIR(误检)" % gallery_size)
    print("=" * 108)
    print(
        "| Enroll |   Rank-1 |   Rank-5 |      mAP |   Probes | TPIR@1e-4 | FNIR@1e-4 | FPIR@1e-4 |"
    )
    print("|" + "|".join(["-" * 10 for _ in range(8)]) + "|")
    for item in results:
        m1n = item["identification_1n"]
        print(
            "| %6d | %7.2f%% | %7.2f%% | %7.2f%% | %8d |%s |%s |%s |"
            % (
                item["enroll_count"],
                m1n["rank1"],
                m1n["rank5"],
                m1n["map"],
                item["probe_images"],
                _fmt_pct(m1n["tpir_at_fpir"].get("1e-4")),
                _fmt_pct(m1n["fnir_at_fpir"].get("1e-4")),
                _fmt_pct(m1n["fpir_at_fpir"].get("1e-4")),
            )
        )
    print("=" * 108)

    print("")
    print("=" * 108)
    print("1:1 验证 — TAR@FAR / FRR@FAR(拒识) / EER")
    print("=" * 108)
    print(
        "| Enroll |  Pos Pairs |  Neg Pairs |  TAR@1e-4 |  FRR@1e-4 |  TAR@1e-3 |  FRR@1e-3 |     EER |"
    )
    print("|" + "|".join(["-" * 12 for _ in range(8)]) + "|")
    for item in results:
        v11 = item["verification_11"]
        print(
            "| %6d | %10d | %10d |%s |%s |%s |%s |%s |"
            % (
                item["enroll_count"],
                v11["positive_pairs"],
                v11["negative_pairs"],
                _fmt_pct(v11["tar_at_far"].get("1e-4")),
                _fmt_pct(v11["frr_at_far"].get("1e-4")),
                _fmt_pct(v11["tar_at_far"].get("1e-3")),
                _fmt_pct(v11["frr_at_far"].get("1e-3")),
                _fmt_pct(v11["eer"]),
            )
        )
    print("=" * 108)


def main():
    parser = argparse.ArgumentParser(
        description="FaceScrub 多注册张数 1:N + 1:1 对比实验"
    )
    parser.add_argument(
        "--megaface-root",
        default="/home/cmsr/桌面/东风/数据集/megaface",
    )
    parser.add_argument("--algo", default="buffalo_l")
    parser.add_argument("--gallery-size", type=int, default=1000000)
    parser.add_argument("--enroll-counts", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--force-reload-distractors",
        action="store_true",
        help="忽略 distractor npz 缓存，从 bin 文件重新加载",
    )
    args = parser.parse_args()

    data_dir = os.path.join(args.megaface_root, "data")
    feature_root = os.path.join(args.megaface_root, "feature_out_clean", args.algo)
    facescrub_root = os.path.join(data_dir, "facescrub_images")
    megaface_lst = os.path.join(data_dir, "megaface_lst")
    noise_path = os.path.join(data_dir, "facescrub_noises.txt")

    if not os.path.isdir(feature_root):
        raise SystemExit(
            "Feature root not found: %s\nRun recognition/_evaluation_/megaface/run_buffalo_l.sh first."
            % feature_root
        )

    noise_names = load_noise_names(noise_path)
    identities = collect_facescrub_by_identity(facescrub_root, noise_names)
    print("FaceScrub identities: %d" % len(identities))

    print("Loading FaceScrub features...")
    t0 = time.time()
    feat_cache = load_facescrub_features(feature_root, identities, args.algo)
    print("Loaded %d FaceScrub features in %.1fs" % (len(feat_cache), time.time() - t0))

    print("Loading MegaFace distractors (gallery_size=%d)..." % args.gallery_size)
    t1 = time.time()
    distractor_feats = load_distractor_features(
        feature_root,
        megaface_lst,
        args.algo,
        args.gallery_size,
        force_reload=args.force_reload_distractors,
    )
    print(
        "Loaded %d distractor features in %.1fs"
        % (distractor_feats.shape[0], time.time() - t1)
    )

    results = []
    for k in args.enroll_counts:
        print("\n>>> enroll_count=%d" % k)
        tk = time.time()
        item = evaluate_enroll_count(k, identities, feat_cache, distractor_feats)
        item["elapsed_sec"] = time.time() - tk
        item["gallery_size"] = distractor_feats.shape[0]
        results.append(item)

        m1n = item["identification_1n"]
        v11 = item["verification_11"]
        print(
            "  1:N Rank-1=%.2f%% Rank-5=%.2f%% mAP=%.2f%%  probes=%d  elapsed=%.1fs"
            % (m1n["rank1"], m1n["rank5"], m1n["map"], item["probe_images"], item["elapsed_sec"])
        )
        for fpir in (1e-2, 1e-3, 1e-4):
            tpir = m1n["tpir_at_fpir"].get(far_label(fpir))
            fnir = m1n["fnir_at_fpir"].get(far_label(fpir))
            if tpir is not None:
                print(
                    "  1:N TPIR@FPIR=%g: %.2f%%  FNIR(漏检)=%.2f%%"
                    % (fpir, tpir, fnir)
                )
        print(
            "  1:1 pos=%d neg=%d  EER=%.2f%%"
            % (v11["positive_pairs"], v11["negative_pairs"], v11["eer"] or 0.0)
        )
        for far in (1e-2, 1e-3, 1e-4):
            tar = v11["tar_at_far"].get(far_label(far))
            frr = v11["frr_at_far"].get(far_label(far))
            if tar is not None:
                print(
                    "  1:1 TAR@FAR=%g: %.2f%%  FRR(拒识)=%.2f%%"
                    % (far, tar, frr)
                )

    print_summary_table(results, distractor_feats.shape[0])

    out_dir = os.path.join(os.path.dirname(__file__), "results", args.algo)
    out_path = args.output or os.path.join(out_dir, "enroll_ablation.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "algo": args.algo,
        "megaface_root": args.megaface_root,
        "gallery_size": distractor_feats.shape[0],
        "enroll_counts": args.enroll_counts,
        "protocol": {
            "dataset": "FaceScrub identities + MegaFace distractors",
            "split": "sorted filename: first k -> enroll, remainder -> probe",
            "fusion": "max similarity per identity across enroll templates",
            "identification_1n": {
                "metrics": [
                    "Rank-1", "Rank-5", "mAP",
                    "TPIR@FPIR", "FNIR(漏检率)", "FPIR(误检率)",
                ],
                "rank1": "correct identity global rank 1 in gallery (identities + distractors)",
                "tpir_at_fpir": {
                    "positive": "probe vs correct identity max score",
                    "negative": "max(wrong identity max score, distractor max score)",
                },
                "fpir_targets": list(FPIR_TARGETS_1N),
            },
            "verification_11": {
                "metrics": [
                    "TAR@FAR", "FRR@FAR(拒识率)", "EER", "best_threshold_accuracy",
                ],
                "positive": "probe vs correct identity max enroll similarity (1 per probe)",
                "negative": "probe vs each wrong identity max enroll similarity (79 per probe)",
                "far_targets": list(FAR_TARGETS_11),
            },
        },
        "results": results,
    }
    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2)
    print("\nSaved: %s" % out_path)


if __name__ == "__main__":
    main()
