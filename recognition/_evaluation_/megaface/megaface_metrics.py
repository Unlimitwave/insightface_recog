from __future__ import print_function

import json
import math
import os
import struct

import numpy as np
from sklearn.metrics import average_precision_score


def load_feature_bin(path, dim=512):
    with open(path, 'rb') as f:
        header = struct.unpack('4i', f.read(16))
        feat = struct.unpack('%df' % header[0], f.read(header[0] * 4))
    return np.asarray(feat[:dim], dtype=np.float32)


def identity_from_facescrub_path(path):
    return path.split('/')[0]


def linear_interp_logx(x_t, x1, x2, y1, y2):
    if x2 <= x1:
        return y1
    if x1 <= 0 or x2 <= 0:
        return y1
    lx, lx1, lx2 = math.log10(x_t), math.log10(x1), math.log10(x2)
    f1 = (lx - lx1) / (lx2 - lx1)
    f2 = 1.0 - f1
    return y1 * f2 + y2 * f1


def nearest_roc_value(roc, target_far):
    fars = roc[0]
    vals = roc[1]
    if not fars:
        return None, None
    if target_far <= fars[0]:
        return vals[0], fars[0]
    for i in range(1, len(fars)):
        if fars[i] >= target_far:
            val = linear_interp_logx(target_far, fars[i - 1], fars[i],
                                     vals[i - 1], vals[i])
            return val, target_far
    return vals[-1], fars[-1]


def parse_cmc_result(data):
    metrics = {
        'rank1': data['cmc'][1][0] * 100.0,
        'traditional_rank1': None,
        'tar_at_far': {},
    }
    if 'traditional_cmc' in data:
        metrics['traditional_rank1'] = data['traditional_cmc'][1][0] * 100.0
    if 'roc' in data:
        for far in (1e-4, 1e-5, 1e-6):
            val, matched_far = nearest_roc_value(data['roc'], far)
            if val is not None:
                metrics['tar_at_far'][far] = {
                    'tar': val * 100.0,
                    'matched_far': matched_far,
                }
    return metrics


def _feature_path(feature_root, rel_path, algo, dataset):
    if dataset == 'facescrub':
        person, name = rel_path.split('/')
        return os.path.join(feature_root, 'facescrub', person,
                            '%s_%s.bin' % (name, algo))
    parts = rel_path.split('/')
    person1, person2, name = parts[-3], parts[-2], parts[-1]
    return os.path.join(feature_root, 'megaface', person1, person2,
                        '%s_%s.bin' % (name, algo))


def compute_map(feature_root, facescrub_list_path, megaface_template_path,
                algo, gallery_size, progress_cb=None):
    with open(facescrub_list_path, 'r') as fp:
        facescrub_paths = [line.strip() for line in fp if line.strip()]
    with open(megaface_template_path, 'r') as fp:
        megaface_paths = json.load(fp)['path']

    gallery_mega = megaface_paths[:gallery_size]
    gallery_paths = list(gallery_mega) + facescrub_paths
    gallery_ids = [''] * len(gallery_mega)
    gallery_ids.extend(identity_from_facescrub_path(p) for p in facescrub_paths)

    gallery_feats = []
    for idx, rel_path in enumerate(gallery_paths):
        if idx < len(gallery_mega):
            feat_path = _feature_path(feature_root, rel_path, algo, 'megaface')
        else:
            feat_path = _feature_path(feature_root, rel_path, algo, 'facescrub')
        if not os.path.isfile(feat_path):
            raise IOError('missing gallery feature: %s' % feat_path)
        gallery_feats.append(load_feature_bin(feat_path))
    gallery_feats = np.stack(gallery_feats, axis=0)
    gallery_feats = gallery_feats / np.linalg.norm(gallery_feats, axis=1, keepdims=True)

    aps = []
    for idx, probe_path in enumerate(facescrub_paths):
        probe_feat_path = _feature_path(feature_root, probe_path, algo, 'facescrub')
        if not os.path.isfile(probe_feat_path):
            continue
        probe_feat = load_feature_bin(probe_feat_path)
        probe_feat = probe_feat / max(np.linalg.norm(probe_feat), 1e-12)
        probe_id = identity_from_facescrub_path(probe_path)

        scores = gallery_feats.dot(probe_feat)
        labels = np.array([1 if gid == probe_id else 0 for gid in gallery_ids],
                          dtype=np.int32)
        if labels.sum() == 0:
            continue
        aps.append(average_precision_score(labels, scores))

        if progress_cb is not None:
            progress_cb(idx + 1, len(facescrub_paths))

    if not aps:
        return None
    return float(np.mean(aps) * 100.0)
