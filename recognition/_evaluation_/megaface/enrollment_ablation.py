#!/usr/bin/env python3
"""Enrollment count ablation on MegaFace FaceScrub with ONNX feature extraction."""

from __future__ import absolute_import, division, print_function

import os
os.environ['ORT_LOGGING_LEVEL'] = '3'

import argparse
import csv
import glob
import json
import math
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import cv2
import numpy as np
import onnxruntime as ort
import sklearn.preprocessing
from insightface.model_zoo import ArcFaceONNX
ort.set_default_logger_severity(3)

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
FPIR_TARGETS = (1e-2, 1e-3, 1e-4)
FAR_TARGETS = (1e-2, 1e-3, 1e-4)


def setup_nvidia_runtime():
    try:
        import site
        nvidia_lib_paths = glob.glob(site.getsitepackages()[0] + '/nvidia/*/lib')
        if nvidia_lib_paths:
            prefix = ':'.join(nvidia_lib_paths)
            ld = os.environ.get('LD_LIBRARY_PATH', '')
            os.environ['LD_LIBRARY_PATH'] = prefix + ((':' + ld) if ld else '')
    except Exception:
        pass


def read_img(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)


def load_model(model_file, gpu):
    available = set(ort.get_available_providers())
    if gpu >= 0 and 'CUDAExecutionProvider' in available:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
        if gpu >= 0:
            print('CUDAExecutionProvider unavailable, fallback to CPU')
            print('available providers:', sorted(available))
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    session = ort.InferenceSession(
        model_file, sess_options=sess_options, providers=providers)
    active = session.get_providers()
    print('session providers:', active)
    if gpu >= 0 and active[0] != 'CUDAExecutionProvider':
        print('Warning: requested GPU but session is using', active[0])
    return ArcFaceONNX(model_file=model_file, session=session)


def get_feature(imgs, model):
    batch_imgs = []
    for img in imgs:
        batch_imgs.append(img)
        batch_imgs.append(cv2.flip(img, 1))
    feats = model.get_feat(batch_imgs)
    count = len(imgs)
    feats = feats.reshape(count, 2, -1)
    embedding = feats[:, 0, :] + feats[:, 1, :]
    embedding = sklearn.preprocessing.normalize(embedding)
    return embedding


def extract_features(paths, model, batch_size=32):
    feats = []
    total = len(paths)
    start = time.time()
    for start_idx in range(0, total, batch_size):
        batch_paths = paths[start_idx:start_idx + batch_size]
        imgs = []
        valid_idx = []
        for j, path in enumerate(batch_paths):
            img = read_img(path)
            if img is None:
                print('read error:', path)
                continue
            imgs.append(img)
            valid_idx.append(start_idx + j)
        if not imgs:
            continue
        batch_feats = get_feature(imgs, model)
        feats.append((valid_idx, batch_feats))
        done = min(start_idx + batch_size, total)
        if done == total or done % 200 == 0:
            elapsed = max(time.time() - start, 1e-6)
            print('feature extraction: %d/%d (%.1f img/s)' % (
                done, total, done / elapsed))
    if not feats:
        return np.zeros((0, 512), dtype=np.float32)
    dim = feats[0][1].shape[1]
    out = np.zeros((total, dim), dtype=np.float32)
    for idx_list, batch_feats in feats:
        for local_i, global_i in enumerate(idx_list):
            out[global_i] = batch_feats[local_i]
    return out


def list_identity_images(facescrub_root, min_images_per_id, max_identities):
    identities = []
    for name in sorted(os.listdir(facescrub_root)):
        person_dir = os.path.join(facescrub_root, name)
        if not os.path.isdir(person_dir):
            continue
        images = sorted(
            f for f in os.listdir(person_dir)
            if os.path.splitext(f)[1] in IMAGE_EXTS)
        if len(images) < min_images_per_id:
            continue
        paths = [os.path.join(person_dir, f) for f in images]
        identities.append((name, paths))
        if max_identities > 0 and len(identities) >= max_identities:
            break
    return identities


def load_distractor_paths(megaface_lst, megaface_images_root, distractor_size):
    paths = []
    with open(megaface_lst, 'r') as fp:
        for line in fp:
            rel = line.strip()
            if not rel:
                continue
            paths.append(os.path.join(megaface_images_root, rel))
            if len(paths) >= distractor_size:
                break
    return paths


def far_label(value):
    mantissa, exponent = f'{value:.0e}'.split('e')
    return '%se%s' % (mantissa, int(exponent))


def tar_at_far(positive_scores, negative_scores, far_targets):
    results = {}
    if not positive_scores or not negative_scores:
        for far in far_targets:
            results[far] = None
        return results

    positives = np.sort(np.asarray(positive_scores, dtype=np.float64))
    negatives = np.sort(np.asarray(negative_scores, dtype=np.float64))
    positive_total = len(positives)
    negative_total = len(negatives)
    all_scores = np.asarray(positive_scores + negative_scores, dtype=np.float64)
    max_negative = float(negatives[-1])
    candidates = np.unique(
        np.concatenate(([np.nextafter(max_negative, np.inf)], all_scores)))
    positive_accepts = (
        positive_total - np.searchsorted(positives, candidates, side='left'))
    false_accepts = (
        negative_total - np.searchsorted(negatives, candidates, side='left'))

    for far in far_targets:
        allowed_fp = int(math.floor(far * negative_total))
        if allowed_fp <= 0:
            results[far] = None
            continue
        valid = false_accepts <= allowed_fp
        if not np.any(valid):
            results[far] = 0.0
            continue
        valid_positive_accepts = positive_accepts[valid]
        best_accepts = int(np.max(valid_positive_accepts))
        best_mask = valid & (positive_accepts == best_accepts)
        best_threshold = float(np.min(candidates[best_mask]))
        results[far] = best_accepts / max(1, positive_total)
    return results


def tpir_at_fpir(pos_scores, neg_score_lists, fpir_targets):
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_all = np.concatenate(neg_score_lists).astype(np.float64)
    correct_cond = pos_scores > np.array([ns.max() for ns in neg_score_lists])
    neg_sorted = np.sort(neg_all)[::-1]
    results = {}
    for fpir in fpir_targets:
        if len(neg_sorted) == 0:
            results[fpir] = None
            continue
        idx = max(int(fpir * len(neg_sorted)) - 1, 0)
        thresh = neg_sorted[idx]
        tpir = np.logical_and(correct_cond, pos_scores >= thresh).sum()
        results[fpir] = float(tpir / max(1, len(pos_scores)))
    return results


def evaluate_enroll_count(k, identity_names, enroll_feats, probe_feats,
                          probe_identity_idx, distractor_feats):
    num_ids = len(identity_names)
    num_probes = probe_feats.shape[0]
    enroll_by_id = {}
    for identity_idx in range(num_ids):
        enroll_by_id[identity_idx] = enroll_feats[identity_idx, :k, :]

    top1 = 0
    top5 = 0
    pos_scores = []
    neg_score_lists = []
    verify_pos = []
    verify_neg = []

    for probe_i in range(num_probes):
        probe_feat = probe_feats[probe_i]
        probe_id = probe_identity_idx[probe_i]

        id_scores = []
        for identity_idx in range(num_ids):
            enroll = enroll_by_id[identity_idx]
            score = float(np.max(enroll @ probe_feat))
            id_scores.append(score)

        if distractor_feats.shape[0] > 0:
            dist_scores = distractor_feats @ probe_feat
        else:
            dist_scores = np.array([], dtype=np.float32)

        gallery_entries = []
        for identity_idx, score in enumerate(id_scores):
            gallery_entries.append((score, ('id', identity_idx)))
        for dist_i, score in enumerate(dist_scores):
            gallery_entries.append((float(score), ('dist', dist_i)))
        gallery_entries.sort(key=lambda item: item[0], reverse=True)

        if gallery_entries and gallery_entries[0][1] == ('id', probe_id):
            top1 += 1
        top5_ids = [entry[1][1] for entry in gallery_entries[:5]
                    if entry[1][0] == 'id']
        if probe_id in top5_ids:
            top5 += 1

        pos_score = id_scores[probe_id]
        neg_scores = [id_scores[j] for j in range(num_ids) if j != probe_id]
        neg_scores.extend(dist_scores.tolist())
        pos_scores.append(pos_score)
        neg_score_lists.append(np.asarray(neg_scores, dtype=np.float32))

        verify_pos.append(pos_score)
        for identity_idx in range(num_ids):
            if identity_idx == probe_id:
                continue
            verify_neg.append(id_scores[identity_idx])

    metrics_1n = {
        'top1': 100.0 * top1 / max(1, num_probes),
        'top5': 100.0 * top5 / max(1, num_probes),
        'num_probes': int(num_probes),
        'num_gallery_identities': int(num_ids),
        'num_distractors': int(distractor_feats.shape[0]),
    }
    for fpir, tpir in tpir_at_fpir(pos_scores, neg_score_lists, FPIR_TARGETS).items():
        key = 'tpir_at_fpir_%s' % far_label(fpir)
        metrics_1n[key] = None if tpir is None else 100.0 * tpir

    metrics_11 = {}
    for far, tar in tar_at_far(verify_pos, verify_neg, FAR_TARGETS).items():
        key = 'tar_at_far_%s' % far_label(far)
        metrics_11[key] = None if tar is None else 100.0 * tar
    metrics_11['num_positive_pairs'] = len(verify_pos)
    metrics_11['num_negative_pairs'] = len(verify_neg)

    return metrics_1n, metrics_11


def print_metrics(k, metrics_1n, metrics_11):
    print('-' * 72)
    print('enroll_count = %d' % k)
    print('1:N  Top-1: %.4f%%  Top-5: %.4f%%' % (
        metrics_1n['top1'], metrics_1n['top5']))
    for fpir in FPIR_TARGETS:
        key = 'tpir_at_fpir_%s' % far_label(fpir)
        val = metrics_1n.get(key)
        if val is None:
            print('1:N  TPIR@FPIR=%g: N/A' % fpir)
        else:
            print('1:N  TPIR@FPIR=%g: %.4f%%' % (fpir, val))
    for far in FAR_TARGETS:
        key = 'tar_at_far_%s' % far_label(far)
        val = metrics_11.get(key)
        if val is None:
            print('1:1  TAR@FAR=%g: N/A' % far)
        else:
            print('1:1  TAR@FAR=%g: %.4f%%' % (far, val))


def save_results(output_dir, all_results, args):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'enrollment_ablation.json')
    csv_path = os.path.join(output_dir, 'enrollment_ablation.csv')

    payload = {
        'config': vars(args),
        'results': all_results,
    }
    with open(json_path, 'w') as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
    print('saved json:', json_path)

    fieldnames = ['enroll_count', 'top1', 'top5']
    for fpir in FPIR_TARGETS:
        fieldnames.append('tpir_at_fpir_%s' % far_label(fpir))
    for far in FAR_TARGETS:
        fieldnames.append('tar_at_far_%s' % far_label(far))
    fieldnames.extend(['num_probes', 'num_positive_pairs', 'num_negative_pairs'])

    with open(csv_path, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for item in all_results:
            row = {
                'enroll_count': item['enroll_count'],
                'top1': item['metrics_1n']['top1'],
                'top5': item['metrics_1n']['top5'],
                'num_probes': item['metrics_1n']['num_probes'],
                'num_positive_pairs': item['metrics_11']['num_positive_pairs'],
                'num_negative_pairs': item['metrics_11']['num_negative_pairs'],
            }
            for fpir in FPIR_TARGETS:
                key = 'tpir_at_fpir_%s' % far_label(fpir)
                row[key] = item['metrics_1n'].get(key)
            for far in FAR_TARGETS:
                key = 'tar_at_far_%s' % far_label(far)
                row[key] = item['metrics_11'].get(key)
            writer.writerow(row)
    print('saved csv:', csv_path)


def parse_enroll_counts(text):
    values = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError('empty --enroll-counts')
    return sorted(set(values))


def main(argv):
    setup_nvidia_runtime()

    default_root = '/home/cmsr/桌面/东风/数据集/megaface'
    default_model = os.path.expanduser(
        '~/.insightface/models/buffalo_l/w600k_r50.onnx')
    parser = argparse.ArgumentParser(
        description='Enrollment count ablation on MegaFace FaceScrub')
    parser.add_argument('--megaface-root', type=str, default=default_root)
    parser.add_argument('--model-file', type=str, default=default_model)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--enroll-counts', type=str, default='1,3,5,10')
    parser.add_argument('--distractor-size', type=int, default=10000)
    parser.add_argument('--min-images-per-id', type=int, default=11)
    parser.add_argument('--max-identities', type=int, default=0)
    parser.add_argument('--output-dir', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args(argv)

    enroll_counts = parse_enroll_counts(args.enroll_counts)
    max_k = max(enroll_counts)
    data_dir = os.path.join(args.megaface_root, 'data')
    facescrub_root = os.path.join(data_dir, 'facescrub_images')
    megaface_images_root = os.path.join(data_dir, 'megaface_images')
    megaface_lst = os.path.join(data_dir, 'megaface_lst')
    output_dir = args.output_dir or os.path.join(
        args.megaface_root, 'results', 'enrollment_ablation')

    if not os.path.isfile(args.model_file):
        raise IOError('model file not found: %s' % args.model_file)
    if not os.path.isdir(facescrub_root):
        raise IOError('facescrub images not found: %s' % facescrub_root)
    if not os.path.isfile(megaface_lst):
        raise IOError('megaface list not found: %s' % megaface_lst)

    print('enrollment ablation')
    print('megaface root:', args.megaface_root)
    print('model file:', args.model_file)
    print('enroll counts:', enroll_counts)
    print('distractor size:', args.distractor_size)
    print('min images per id:', args.min_images_per_id)
    print('max identities:', args.max_identities or 'all')
    print('output dir:', output_dir)

    identities = list_identity_images(
        facescrub_root, args.min_images_per_id, args.max_identities)
    if not identities:
        raise SystemExit('no identities satisfy min-images-per-id')

    identity_names = [item[0] for item in identities]
    identity_paths = [item[1] for item in identities]
    print('identities used:', len(identity_names))

    max_images = max(len(paths) for paths in identity_paths)
    if max_images <= max_k:
        raise SystemExit(
            'max enroll count %d requires more than %d images per identity' % (
                max_k, max_images))

    flat_paths = []
    path_owner = []
    enroll_counts_per_id = []
    probe_counts_per_id = []
    for identity_idx, paths in enumerate(identity_paths):
        flat_paths.extend(paths)
        path_owner.extend([identity_idx] * len(paths))
        enroll_counts_per_id.append(min(max_k, len(paths)))
        probe_counts_per_id.append(max(0, len(paths) - max_k))

    distractor_paths = load_distractor_paths(
        megaface_lst, megaface_images_root, args.distractor_size)
    print('distractor images:', len(distractor_paths))

    all_paths = flat_paths + distractor_paths
    print('total images to extract:', len(all_paths))

    model = load_model(args.model_file, args.gpu)
    all_feats = extract_features(all_paths, model, batch_size=args.batch_size)
    facescrub_feats = all_feats[:len(flat_paths)]
    distractor_feats = all_feats[len(flat_paths):]

    max_enroll_feats = np.zeros(
        (len(identity_names), max_k, facescrub_feats.shape[1]), dtype=np.float32)
    probe_feats_list = []
    probe_identity_idx = []
    offset = 0
    for identity_idx, paths in enumerate(identity_paths):
        count = len(paths)
        feats = facescrub_feats[offset:offset + count]
        offset += count
        max_enroll_feats[identity_idx, :min(count, max_k), :] = feats[:max_k]
        if count > max_k:
            probe_feats_list.append(feats[max_k:])
            probe_identity_idx.extend([identity_idx] * (count - max_k))

    if not probe_feats_list:
        raise SystemExit('no probe images after enrollment split')
    probe_feats = np.vstack(probe_feats_list)
    probe_identity_idx = np.asarray(probe_identity_idx, dtype=np.int32)
    print('probe images:', probe_feats.shape[0])

    all_results = []
    for k in enroll_counts:
        metrics_1n, metrics_11 = evaluate_enroll_count(
            k, identity_names, max_enroll_feats, probe_feats,
            probe_identity_idx, distractor_feats)
        print_metrics(k, metrics_1n, metrics_11)
        all_results.append({
            'enroll_count': k,
            'metrics_1n': metrics_1n,
            'metrics_11': metrics_11,
        })

    save_results(output_dir, all_results, args)
    print('done')


if __name__ == '__main__':
    main(sys.argv[1:])
