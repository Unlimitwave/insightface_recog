from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Must be set before importing onnxruntime (C++ logger reads this at import time).
import os
os.environ['ORT_LOGGING_LEVEL'] = '3'

import argparse
import struct
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


def read_img(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)


def write_bin(path, feature):
    feature = list(feature.astype(np.float32))
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', len(feature), 1, 4, 5))
        f.write(struct.pack('%df' % len(feature), *feature))


def count_lines(path):
    count = 0
    with open(path, 'r') as fp:
        for line in fp:
            if line.strip():
                count += 1
    return count


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


def get_and_write(buffer, model):
    imgs = [item[0] for item in buffer]
    features = get_feature(imgs, model)
    assert features.shape[0] == len(buffer)
    for ik, item in enumerate(buffer):
        write_bin(item[1], features[ik].flatten())


def print_progress(list_name, processed, total, written, skipped, start_time):
    elapsed = max(time.time() - start_time, 1e-6)
    pct = 100.0 * processed / max(total, 1)
    rate = written / elapsed
    remain = max(total - processed, 0)
    eta_sec = remain / max(rate, 1e-6) if written > 0 else 0.0
    print('%s progress: %d/%d (%.1f%%) written=%d skipped=%d rate=%.1f img/s eta=%.0fs' % (
        list_name, processed, total, pct, written, skipped, rate, eta_sec))


def process_image_list(lst_path, image_root, out_root, rel_depth, model, algo,
                       batch_size, skip_existing, list_name):
    total = count_lines(lst_path)
    print('%s total images: %d' % (list_name, total))
    processed = 0
    written = 0
    skipped = 0
    buffer = []
    start_time = time.time()
    last_report = 0

    for line in open(lst_path, 'r'):
        image_path = line.strip()
        if not image_path:
            continue
        processed += 1
        parts = image_path.split('/')
        if rel_depth == 2:
            a, b = parts[-2], parts[-1]
            out_dir = os.path.join(out_root, a)
            out_name = b
        elif rel_depth == 3:
            a1, a2, b = parts[-3], parts[-2], parts[-1]
            out_dir = os.path.join(out_root, a1, a2)
            out_name = b
        else:
            raise ValueError('unsupported rel_depth: %s' % rel_depth)

        out_path = os.path.join(out_dir, '%s_%s.bin' % (out_name, algo))
        if skip_existing and os.path.isfile(out_path):
            skipped += 1
            if processed - last_report >= 1000 or processed == total:
                print_progress(list_name, processed, total, written, skipped, start_time)
                last_report = processed
            continue

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        full_image_path = os.path.join(image_root, image_path)
        img = read_img(full_image_path)
        if img is None:
            print('read error:', full_image_path)
            continue

        buffer.append((img, out_path))
        if len(buffer) == batch_size:
            get_and_write(buffer, model)
            written += len(buffer)
            buffer = []

        if processed - last_report >= 100 or processed == total:
            print_progress(list_name, processed, total, written, skipped, start_time)
            last_report = processed

    if len(buffer) > 0:
        get_and_write(buffer, model)
        written += len(buffer)

    print('%s done total=%d written=%d skipped=%d elapsed=%.1fs' % (
        list_name, total, written, skipped, time.time() - start_time))


def main(args):
    print(args)
    if not os.path.isfile(args.model_file):
        raise IOError('model file not found: %s' % args.model_file)

    model = load_model(args.model_file, args.gpu)
    facescrub_out = os.path.join(args.output, 'facescrub')
    megaface_out = os.path.join(args.output, 'megaface')

    if not args.skip_facescrub:
        process_image_list(
            args.facescrub_lst, args.facescrub_root, facescrub_out, 2, model,
            args.algo, args.batch_size, args.skip_existing, 'facescrub')

    if not args.skip_megaface:
        process_image_list(
            args.megaface_lst, args.megaface_root, megaface_out, 3, model,
            args.algo, args.batch_size, args.skip_existing, 'megaface')


def parse_arguments(argv):
    default_model = os.path.expanduser(
        '~/.insightface/models/buffalo_l/w600k_r50.onnx')
    parser = argparse.ArgumentParser(
        description='Extract MegaFace features with InsightFace ONNX models')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id, set -1 for CPU')
    parser.add_argument('--algo', type=str, default='buffalo_l',
                        help='suffix used in feature .bin filenames')
    parser.add_argument('--model-file', type=str, default=default_model,
                        help='path to ONNX recognition model')
    parser.add_argument('--facescrub-lst', type=str,
                        default='./data/facescrub_lst')
    parser.add_argument('--megaface-lst', type=str,
                        default='./data/megaface_lst')
    parser.add_argument('--facescrub-root', type=str,
                        default='./data/facescrub_images')
    parser.add_argument('--megaface-root', type=str,
                        default='./data/megaface_images')
    parser.add_argument('--output', type=str, default='./feature_out')
    parser.add_argument('--skip-existing', action='store_true',
                        help='skip images whose feature file already exists')
    parser.add_argument('--skip-facescrub', action='store_true')
    parser.add_argument('--skip-megaface', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
