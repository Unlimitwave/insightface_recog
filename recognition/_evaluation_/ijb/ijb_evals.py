#!/usr/bin/env python3
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import transform
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, auc

_EVAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _EVAL_ROOT not in sys.path:
    sys.path.insert(0, _EVAL_ROOT)

from metrics.production_metrics import (
    FPIR_TARGETS_1N_DISPLAY,
    FAR_TARGETS_11_DISPLAY,
    compute_map_from_ranks,
    far_label,
    format_identification_table,
    format_verification_table,
    summarize_identification_metrics,
    summarize_verification_metrics,
    summarize_verification_metrics_from_roc,
)


def _log_stage(msg):
    import time

    print(">>>> [%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


class Mxnet_model_interf:
    def __init__(self, model_file, layer="fc1", image_size=(112, 112)):
        import mxnet as mx

        self.mx = mx
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if len(cvd) > 0 and int(cvd) != -1:
            ctx = [self.mx.gpu(ii) for ii in range(len(cvd.split(",")))]
        else:
            ctx = [self.mx.cpu()]

        prefix, epoch = model_file.split(",")
        print(">>>> loading mxnet model:", prefix, epoch, ctx)
        sym, arg_params, aux_params = self.mx.model.load_checkpoint(prefix, int(epoch))
        all_layers = sym.get_internals()
        sym = all_layers[layer + "_output"]
        model = self.mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(data_shapes=[("data", (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model

    def __call__(self, imgs):
        # print(imgs.shape, imgs[0])
        imgs = imgs.transpose(0, 3, 1, 2)
        data = self.mx.nd.array(imgs)
        db = self.mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        emb = self.model.get_outputs()[0].asnumpy()
        return emb


class Torch_model_interf:
    def __init__(self, model_file, image_size=(112, 112)):
        import torch

        self.torch = torch
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        device_name = "cuda:0" if len(cvd) > 0 and int(cvd) != -1 else "cpu"
        self.device = self.torch.device(device_name)
        try:
            self.model = self.torch.jit.load(model_file, map_location=device_name)
        except:
            print("Error: %s is weights only, please load and save the entire model by `torch.jit.save`" % model_file)
            self.model = None

    def __call__(self, imgs):
        # print(imgs.shape, imgs[0])
        imgs = imgs.transpose(0, 3, 1, 2).copy().astype("float32")
        imgs = (imgs - 127.5) * 0.0078125
        output = self.model(self.torch.from_numpy(imgs).to(self.device).float())
        return output.cpu().detach().numpy()


class ONNX_model_interf:
    def __init__(self, model_file, image_size=(112, 112)):
        import onnxruntime as ort
        import glob
        import site

        # Try to load CUDA deps shipped via conda `nvidia-*` packages.
        # This mirrors `recognition/_evaluation_/megaface/run_buffalo_l.sh`.
        try:
            nvidia_lib_paths = glob.glob(site.getsitepackages()[0] + "/nvidia/*/lib")
            if nvidia_lib_paths:
                ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
                prefix = ":".join(nvidia_lib_paths)
                os.environ["LD_LIBRARY_PATH"] = prefix + ((":" + ld_library_path) if ld_library_path else "")
        except Exception:
            pass

        ort.set_default_logger_severity(3)
        # Prefer CUDA when available, otherwise fall back to CPU.
        self.ort_session = ort.InferenceSession(
            model_file, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        print(">>>> ONNXRuntime providers:", self.ort_session.get_providers())
        self.output_names = [self.ort_session.get_outputs()[0].name]
        self.input_name = self.ort_session.get_inputs()[0].name

    def __call__(self, imgs):
        imgs = imgs.transpose(0, 3, 1, 2).astype("float32")
        imgs = (imgs - 127.5) * 0.0078125
        outputs = self.ort_session.run(self.output_names, {self.input_name: imgs})
        return outputs[0]


def keras_model_interf(model_file):
    import tensorflow as tf
    from tensorflow_addons.layers import StochasticDepth

    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    mm = tf.keras.models.load_model(model_file, compile=False)
    return lambda imgs: mm((tf.cast(imgs, "float32") - 127.5) * 0.0078125).numpy()


def face_align_landmark(img, landmark, image_size=(112, 112), method="similar"):
    tform = transform.AffineTransform() if method == "affine" else transform.SimilarityTransform()
    src = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]], dtype=np.float32
    )
    tform.estimate(landmark, src)
    # ndimage = transform.warp(img, tform.inverse, output_shape=image_size)
    # ndimage = (ndimage * 255).astype(np.uint8)
    M = tform.params[0:2, :]
    ndimage = cv2.warpAffine(img, M, image_size, borderValue=0.0)
    if len(ndimage.shape) == 2:
        ndimage = np.stack([ndimage, ndimage, ndimage], -1)
    else:
        ndimage = cv2.cvtColor(ndimage, cv2.COLOR_BGR2RGB)
    return ndimage


def read_IJB_meta_columns_to_int(file_path, columns, sep=" ", skiprows=0, header=None):
    # meta = np.loadtxt(file_path, skiprows=skiprows, delimiter=sep)
    meta = pd.read_csv(file_path, sep=sep, skiprows=skiprows, header=header).values
    return (meta[:, ii].astype("int") for ii in columns)


def extract_IJB_data_11(data_path, subset, save_path=None, force_reload=False):
    if save_path == None:
        save_path = os.path.join(data_path, subset + "_backup.npz")
    if not force_reload and os.path.exists(save_path):
        print(">>>> Reload from backup: %s ..." % save_path)
        aa = np.load(save_path)
        return (
            aa["templates"],
            aa["medias"],
            aa["p1"],
            aa["p2"],
            aa["label"],
            aa["img_names"],
            aa["landmarks"],
            aa["face_scores"],
        )

    if subset == "IJBB":
        media_list_path = os.path.join(data_path, "IJBB/meta/ijbb_face_tid_mid.txt")
        pair_list_path = os.path.join(data_path, "IJBB/meta/ijbb_template_pair_label.txt")
        img_path = os.path.join(data_path, "IJBB/loose_crop")
        img_list_path = os.path.join(data_path, "IJBB/meta/ijbb_name_5pts_score.txt")
    else:
        media_list_path = os.path.join(data_path, "IJBC/meta/ijbc_face_tid_mid.txt")
        pair_list_path = os.path.join(data_path, "IJBC/meta/ijbc_template_pair_label.txt")
        img_path = os.path.join(data_path, "IJBC/loose_crop")
        img_list_path = os.path.join(data_path, "IJBC/meta/ijbc_name_5pts_score.txt")

    print(">>>> Loading templates and medias...")
    templates, medias = read_IJB_meta_columns_to_int(media_list_path, columns=[1, 2])  # ['1.jpg', '1', '69544']
    print("templates: %s, medias: %s, unique templates: %s" % (templates.shape, medias.shape, np.unique(templates).shape))
    # templates: (227630,), medias: (227630,), unique templates: (12115,)

    print(">>>> Loading pairs...")
    p1, p2, label = read_IJB_meta_columns_to_int(pair_list_path, columns=[0, 1, 2])  # ['1', '11065', '1']
    print("p1: %s, unique p1: %s" % (p1.shape, np.unique(p1).shape))
    print("p2: %s, unique p2: %s" % (p2.shape, np.unique(p2).shape))
    print("label: %s, label value counts: %s" % (label.shape, dict(zip(*np.unique(label, return_counts=True)))))
    # p1: (8010270,), unique p1: (1845,)
    # p2: (8010270,), unique p2: (10270,) # 10270 + 1845 = 12115 --> np.unique(templates).shape
    # label: (8010270,), label value counts: {0: 8000000, 1: 10270}

    print(">>>> Loading images...")
    with open(img_list_path, "r") as ff:
        # 1.jpg 46.060 62.026 87.785 60.323 68.851 77.656 52.162 99.875 86.450 98.648 0.999
        img_records = np.array([ii.strip().split(" ") for ii in ff.readlines()])

    img_names = np.array([os.path.join(img_path, ii) for ii in img_records[:, 0]])
    landmarks = img_records[:, 1:-1].astype("float32").reshape(-1, 5, 2)
    face_scores = img_records[:, -1].astype("float32")
    print("img_names: %s, landmarks: %s, face_scores: %s" % (img_names.shape, landmarks.shape, face_scores.shape))
    # img_names: (227630,), landmarks: (227630, 5, 2), face_scores: (227630,)
    print("face_scores value counts:", dict(zip(*np.histogram(face_scores, bins=9)[::-1])))
    # {0.1: 2515, 0.2: 0, 0.3: 62, 0.4: 94, 0.5: 136, 0.6: 197, 0.7: 291, 0.8: 538, 0.9: 223797}

    print(">>>> Saving backup to: %s ..." % save_path)
    np.savez(
        save_path,
        templates=templates,
        medias=medias,
        p1=p1,
        p2=p2,
        label=label,
        img_names=img_names,
        landmarks=landmarks,
        face_scores=face_scores,
    )
    print()
    return templates, medias, p1, p2, label, img_names, landmarks, face_scores


IJB_1N_META_IJBC = (
    "ijbc_1N_gallery_G1.csv",
    "ijbc_1N_gallery_G2.csv",
    "ijbc_1N_probe_mixed.csv",
)
IJB_1N_META_IJBB = (
    "ijbb_1N_gallery_S1.csv",
    "ijbb_1N_gallery_S2.csv",
    "ijbb_1N_probe_mixed.csv",
)


def get_ijb_1n_meta_paths(data_path, subset):
    if subset == "IJBC":
        meta_dir = os.path.join(data_path, "IJBC", "meta")
        names = IJB_1N_META_IJBC
    else:
        meta_dir = os.path.join(data_path, "IJBB", "meta")
        names = IJB_1N_META_IJBB
    return {name: os.path.join(meta_dir, name) for name in names}


def check_ijb_1n_meta(data_path, subset):
    """Ensure 1:N protocol CSVs exist before running hours-long embedding."""
    paths = get_ijb_1n_meta_paths(data_path, subset)
    missing = [name for name, path in paths.items() if not os.path.isfile(path)]
    if not missing:
        return paths
    meta_dir = os.path.dirname(next(iter(paths.values())))
    present = sorted(os.listdir(meta_dir)) if os.path.isdir(meta_dir) else []
    raise FileNotFoundError(
        "\n"
        "========== IJB 1:N meta files missing ==========\n"
        "Directory: %s\n"
        "Missing (%d):\n  %s\n\n"
        "Present meta files:\n  %s\n\n"
        "1:1 evaluation only needs *_template_pair_label.txt (you already have these).\n"
        "1:N requires 3 extra CSV protocol files — download InsightFace \"Updated Meta (1:1 and 1:N)\":\n"
        "  GDrive: https://drive.google.com/file/d/1MXzrU_zUESSx_242pRUnVvW_wDzfU8Ky/view\n"
        "  Baidu:  https://pan.baidu.com/s/1x-ytzg4zkCTOTtklUgAhfg  (code: 7g8o)\n\n"
        "After download, copy the 3 CSV files into:\n"
        "  %s/\n"
        "Then re-run with -N.\n"
        "See: recognition/_evaluation_/ijb/README.md\n"
        "================================================"
        % (
            meta_dir,
            len(missing),
            "\n  ".join(missing),
            "\n  ".join(present) if present else "(meta dir empty or not found)",
            meta_dir,
        )
    )


def extract_gallery_prob_data(data_path, subset, save_path=None, force_reload=False):
    if save_path == None:
        save_path = os.path.join(data_path, subset + "_gallery_prob_backup.npz")
    if not force_reload and os.path.exists(save_path):
        print(">>>> Reload from backup: %s ..." % save_path)
        aa = np.load(save_path)
        return (
            aa["s1_templates"],
            aa["s1_subject_ids"],
            aa["s2_templates"],
            aa["s2_subject_ids"],
            aa["probe_mixed_templates"],
            aa["probe_mixed_subject_ids"],
        )

    if subset == "IJBC":
        meta_dir = os.path.join(data_path, "IJBC/meta")
        gallery_s1_record = os.path.join(meta_dir, "ijbc_1N_gallery_G1.csv")
        gallery_s2_record = os.path.join(meta_dir, "ijbc_1N_gallery_G2.csv")
        probe_mixed_record = os.path.join(meta_dir, "ijbc_1N_probe_mixed.csv")
    else:
        meta_dir = os.path.join(data_path, "IJBB/meta")
        gallery_s1_record = os.path.join(meta_dir, "ijbb_1N_gallery_S1.csv")
        gallery_s2_record = os.path.join(meta_dir, "ijbb_1N_gallery_S2.csv")
        probe_mixed_record = os.path.join(meta_dir, "ijbb_1N_probe_mixed.csv")

    print(">>>> Loading gallery feature...")
    s1_templates, s1_subject_ids = read_IJB_meta_columns_to_int(gallery_s1_record, columns=[0, 1], skiprows=1, sep=",")
    s2_templates, s2_subject_ids = read_IJB_meta_columns_to_int(gallery_s2_record, columns=[0, 1], skiprows=1, sep=",")
    print("s1 gallery: %s, ids: %s, unique: %s" % (s1_templates.shape, s1_subject_ids.shape, np.unique(s1_templates).shape))
    print("s2 gallery: %s, ids: %s, unique: %s" % (s2_templates.shape, s2_subject_ids.shape, np.unique(s2_templates).shape))

    print(">>>> Loading prope feature...")
    probe_mixed_templates, probe_mixed_subject_ids = read_IJB_meta_columns_to_int(
        probe_mixed_record, columns=[0, 1], skiprows=1, sep=","
    )
    print("probe_mixed_templates: %s, unique: %s" % (probe_mixed_templates.shape, np.unique(probe_mixed_templates).shape))
    print("probe_mixed_subject_ids: %s, unique: %s" % (probe_mixed_subject_ids.shape, np.unique(probe_mixed_subject_ids).shape))

    print(">>>> Saving backup to: %s ..." % save_path)
    np.savez(
        save_path,
        s1_templates=s1_templates,
        s1_subject_ids=s1_subject_ids,
        s2_templates=s2_templates,
        s2_subject_ids=s2_subject_ids,
        probe_mixed_templates=probe_mixed_templates,
        probe_mixed_subject_ids=probe_mixed_subject_ids,
    )
    print()
    return s1_templates, s1_subject_ids, s2_templates, s2_subject_ids, probe_mixed_templates, probe_mixed_subject_ids


def get_embeddings(model_interf, img_names, landmarks, batch_size=64, flip=True):
    steps = int(np.ceil(len(img_names) / batch_size))
    embs, embs_f = [], []
    for batch_id in tqdm(range(0, len(img_names), batch_size), "Embedding", total=steps):
        batch_imgs, batch_landmarks = img_names[batch_id : batch_id + batch_size], landmarks[batch_id : batch_id + batch_size]
        ndimages = [face_align_landmark(cv2.imread(img), landmark) for img, landmark in zip(batch_imgs, batch_landmarks)]
        ndimages = np.stack(ndimages)
        embs.extend(model_interf(ndimages))
        if flip:
            embs_f.extend(model_interf(ndimages[:, :, ::-1, :]))
    return np.array(embs), np.array(embs_f)


def process_embeddings(embs, embs_f=[], use_flip_test=True, use_norm_score=False, use_detector_score=True, face_scores=None):
    print(">>>> process_embeddings: Norm {}, Detect_score {}, Flip {}".format(use_norm_score, use_detector_score, use_flip_test))
    if use_flip_test and len(embs_f) != 0:
        embs = embs + embs_f
    if use_norm_score:
        embs = normalize(embs)
    if use_detector_score and face_scores is not None:
        embs = embs * np.expand_dims(face_scores, -1)
    return embs


def image2template_feature(img_feats=None, templates=None, medias=None, choose_templates=None, choose_ids=None):
    if choose_templates is not None:  # 1:N
        unique_templates, indices = np.unique(choose_templates, return_index=True)
        unique_subjectids = choose_ids[indices]
    else:  # 1:1
        unique_templates = np.unique(templates)
        unique_subjectids = None

    # template_feats = np.zeros((len(unique_templates), img_feats.shape[1]), dtype=img_feats.dtype)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
    for count_template, uqt in tqdm(enumerate(unique_templates), "Extract template feature", total=len(unique_templates)):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [np.mean(face_norm_feats[ind_m], 0, keepdims=True)]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, 0)
    template_norm_feats = normalize(template_feats)
    return template_norm_feats, unique_templates, unique_subjectids


def verification_11(template_norm_feats=None, unique_templates=None, p1=None, p2=None, batch_size=10000):
    try:
        print(">>>> Trying cupy.")
        import cupy as cp

        template_norm_feats = cp.array(template_norm_feats)
        score_func = lambda feat1, feat2: cp.sum(feat1 * feat2, axis=-1).get()
        test = score_func(template_norm_feats[:batch_size], template_norm_feats[:batch_size])
    except:
        score_func = lambda feat1, feat2: np.sum(feat1 * feat2, -1)

    template2id = np.zeros(max(unique_templates) + 1, dtype=int)
    template2id[unique_templates] = np.arange(len(unique_templates))

    steps = int(np.ceil(len(p1) / batch_size))
    score = []
    for id in tqdm(range(steps), "Verification"):
        feat1 = template_norm_feats[template2id[p1[id * batch_size : (id + 1) * batch_size]]]
        feat2 = template_norm_feats[template2id[p2[id * batch_size : (id + 1) * batch_size]]]
        score.extend(score_func(feat1, feat2))
    return np.array(score)


def evaluation_1N(query_feats, gallery_feats, query_ids, reg_ids, fars=[0.01, 0.1]):
    print("query_feats: %s, gallery_feats: %s" % (query_feats.shape, gallery_feats.shape))
    similarity = np.dot(query_feats, gallery_feats.T)  # (19593, 3531)

    top_1_count, top_5_count, top_10_count = 0, 0, 0
    pos_sims, neg_sims, non_gallery_sims = [], [], []
    ranks = []
    for index, query_id in enumerate(query_ids):
        if query_id in reg_ids:
            gallery_label = np.argwhere(reg_ids == query_id)[0, 0]
            index_sorted = np.argsort(similarity[index])[::-1]

            top_1_count += gallery_label in index_sorted[:1]
            top_5_count += gallery_label in index_sorted[:5]
            top_10_count += gallery_label in index_sorted[:10]
            rank = int(np.where(index_sorted == gallery_label)[0][0]) + 1
            ranks.append(rank)

            pos_sims.append(similarity[index][reg_ids == query_id][0])
            neg_sims.append(similarity[index][reg_ids != query_id])
        else:
            non_gallery_sims.append(similarity[index])
    total_pos = len(pos_sims)
    pos_sims, neg_sims, non_gallery_sims = np.array(pos_sims), np.array(neg_sims), np.array(non_gallery_sims)
    print("pos_sims: %s, neg_sims: %s, non_gallery_sims: %s" % (pos_sims.shape, neg_sims.shape, non_gallery_sims.shape))

    rank1 = top_1_count / total_pos
    rank5 = top_5_count / total_pos
    rank10 = top_10_count / total_pos
    map_score = compute_map_from_ranks(ranks)
    print(
        "Rank-1: %f, Rank-5: %f, Rank-10: %f, mAP: %f"
        % (rank1, rank5, rank10, map_score)
    )

    correct_pos_cond = pos_sims > neg_sims.max(1)
    non_gallery_max = non_gallery_sims.max(1)
    non_gallery_sims_sorted = np.sort(non_gallery_max)[::-1]
    threshes, recalls, fpirs, fnirs = [], [], [], []
    for far in fars:
        thresh = non_gallery_sims_sorted[max(int((non_gallery_sims_sorted.shape[0]) * far) - 1, 0)]
        tpir = np.logical_and(correct_pos_cond, pos_sims > thresh).sum() / pos_sims.shape[0]
        fpir = float(np.mean(non_gallery_max >= thresh))
        threshes.append(thresh)
        recalls.append(tpir)
        fpirs.append(fpir)
        fnirs.append(1.0 - tpir)
    cmc_scores = list(zip(neg_sims, pos_sims.reshape(-1, 1))) + list(zip(non_gallery_sims, [None] * non_gallery_sims.shape[0]))
    metrics = summarize_identification_metrics(
        rank1=rank1,
        rank5=rank5,
        rank10=rank10,
        map_score=map_score,
        enrolled_probe_count=total_pos,
        tpir_detail={
            far: {
                "tar": recalls[idx],
                "far": fpirs[idx],
                "frr": fnirs[idx],
                "threshold": threshes[idx],
            }
            for idx, far in enumerate(fars)
        },
        fpir_targets=fars,
    )
    return (
        top_1_count,
        top_5_count,
        top_10_count,
        threshes,
        recalls,
        cmc_scores,
        metrics,
    )


class IJB_test:
    def __init__(self, model_file, data_path, subset, batch_size=64, force_reload=False, restore_embs=None):
        templates, medias, p1, p2, label, img_names, landmarks, face_scores = extract_IJB_data_11(
            data_path, subset, force_reload=force_reload
        )
        if model_file != None:
            if model_file.endswith(".h5"):
                interf_func = keras_model_interf(model_file)
            elif model_file.endswith(".pth") or model_file.endswith(".pt"):
                interf_func = Torch_model_interf(model_file)
            elif model_file.endswith(".onnx") or model_file.endswith(".ONNX"):
                interf_func = ONNX_model_interf(model_file)
            else:
                interf_func = Mxnet_model_interf(model_file)
            self.embs, self.embs_f = get_embeddings(interf_func, img_names, landmarks, batch_size=batch_size)
        elif restore_embs != None:
            print(">>>> Reload embeddings from:", restore_embs)
            aa = np.load(restore_embs)
            if "embs" in aa and "embs_f" in aa:
                self.embs, self.embs_f = aa["embs"], aa["embs_f"]
            else:
                print("ERROR: %s NOT containing embs / embs_f" % restore_embs)
                exit(1)
            print(">>>> Done.")
        self.data_path, self.subset, self.force_reload = data_path, subset, force_reload
        self.templates, self.medias, self.p1, self.p2, self.label = templates, medias, p1, p2, label
        self.face_scores = face_scores.astype(self.embs.dtype)

    def run_model_test_single(self, use_flip_test=True, use_norm_score=False, use_detector_score=True):
        img_input_feats = process_embeddings(
            self.embs,
            self.embs_f,
            use_flip_test=use_flip_test,
            use_norm_score=use_norm_score,
            use_detector_score=use_detector_score,
            face_scores=self.face_scores,
        )
        template_norm_feats, unique_templates, _ = image2template_feature(img_input_feats, self.templates, self.medias)
        score = verification_11(template_norm_feats, unique_templates, self.p1, self.p2)
        return score

    def run_model_test_bunch(self):
        from itertools import product

        scores, names = [], []
        for use_norm_score, use_detector_score, use_flip_test in product([True, False], [True, False], [True, False]):
            name = "N{:d}D{:d}F{:d}".format(use_norm_score, use_detector_score, use_flip_test)
            print(">>>>", name, use_norm_score, use_detector_score, use_flip_test)
            names.append(name)
            scores.append(self.run_model_test_single(use_flip_test, use_norm_score, use_detector_score))
        return scores, names

    def run_model_test_1N(self, npoints=100):
        check_ijb_1n_meta(self.data_path, self.subset)
        fars_cal = [10 ** ii for ii in np.arange(-4, 0, 4 / npoints)] + [1]  # plot in range [10-4, 1]
        fars_show_idx = np.arange(len(fars_cal))[:: npoints // 4]  # npoints=100, fars_show=[0.0001, 0.001, 0.01, 0.1, 1.0]

        g1_templates, g1_ids, g2_templates, g2_ids, probe_mixed_templates, probe_mixed_ids = extract_gallery_prob_data(
            self.data_path, self.subset, force_reload=self.force_reload
        )
        img_input_feats = process_embeddings(
            self.embs,
            self.embs_f,
            use_flip_test=True,
            use_norm_score=False,
            use_detector_score=True,
            face_scores=self.face_scores,
        )
        g1_templates_feature, g1_unique_templates, g1_unique_ids = image2template_feature(
            img_input_feats, self.templates, self.medias, g1_templates, g1_ids
        )
        g2_templates_feature, g2_unique_templates, g2_unique_ids = image2template_feature(
            img_input_feats, self.templates, self.medias, g2_templates, g2_ids
        )
        probe_mixed_templates_feature, probe_mixed_unique_templates, probe_mixed_unique_subject_ids = image2template_feature(
            img_input_feats, self.templates, self.medias, probe_mixed_templates, probe_mixed_ids
        )
        print("g1_templates_feature:", g1_templates_feature.shape)  # (1772, 512)
        print("g2_templates_feature:", g2_templates_feature.shape)  # (1759, 512)

        print("probe_mixed_templates_feature:", probe_mixed_templates_feature.shape)  # (19593, 512)
        print("probe_mixed_unique_subject_ids:", probe_mixed_unique_subject_ids.shape)  # (19593,)

        print(">>>> Gallery 1")
        g1_top_1_count, g1_top_5_count, g1_top_10_count, g1_threshes, g1_recalls, g1_cmc_scores, g1_metrics = evaluation_1N(
            probe_mixed_templates_feature, g1_templates_feature, probe_mixed_unique_subject_ids, g1_unique_ids, fars_cal
        )
        print(format_identification_table(g1_metrics, gallery_name="Gallery 1"))
        print(">>>> Gallery 2")
        g2_top_1_count, g2_top_5_count, g2_top_10_count, g2_threshes, g2_recalls, g2_cmc_scores, g2_metrics = evaluation_1N(
            probe_mixed_templates_feature, g2_templates_feature, probe_mixed_unique_subject_ids, g2_unique_ids, fars_cal
        )
        print(format_identification_table(g2_metrics, gallery_name="Gallery 2"))
        print(">>>> Mean")
        query_num = probe_mixed_templates_feature.shape[0]
        top_1 = (g1_top_1_count + g2_top_1_count) / query_num
        top_5 = (g1_top_5_count + g2_top_5_count) / query_num
        top_10 = (g1_top_10_count + g2_top_10_count) / query_num
        map_score = (g1_metrics["map"] + g2_metrics["map"]) / 2.0
        print(
            "[Mean] Rank-1: %f, Rank-5: %f, Rank-10: %f, mAP: %f"
            % (top_1, top_5, top_10, map_score)
        )

        mean_tpirs = (np.array(g1_recalls) + np.array(g2_recalls)) / 2
        mean_fnirs = 1.0 - mean_tpirs
        show_fars = [fars_cal[i] for i in fars_show_idx]
        show_result = {"fpir": show_fars}
        for prefix, recalls, threshes, gallery_metrics in (
            ("g1", g1_recalls, g1_threshes, g1_metrics),
            ("g2", g2_recalls, g2_threshes, g2_metrics),
        ):
            show_result["%s_tpir" % prefix] = [recalls[i] for i in fars_show_idx]
            show_result["%s_fnir" % prefix] = [1.0 - recalls[i] for i in fars_show_idx]
            show_result["%s_fpir" % prefix] = [
                gallery_metrics["tpir_at_fpir_detail"][far_label(fars_cal[i])]["far"]
                for i in fars_show_idx
            ]
            show_result["%s_thresh" % prefix] = [threshes[i] for i in fars_show_idx]
        show_result["mean_tpir"] = [mean_tpirs[i] for i in fars_show_idx]
        show_result["mean_fnir"] = [mean_fnirs[i] for i in fars_show_idx]
        show_result["mean_fpir"] = [
            (
                g1_metrics["tpir_at_fpir_detail"][far_label(fars_cal[i])]["far"]
                + g2_metrics["tpir_at_fpir_detail"][far_label(fars_cal[i])]["far"]
            )
            / 2.0
            for i in fars_show_idx
        ]

        mean_metrics = summarize_identification_metrics(
            rank1=top_1,
            rank5=top_5,
            rank10=top_10,
            map_score=map_score,
            enrolled_probe_count=int(g1_metrics["enrolled_probe_count"]),
            tpir_detail={
                fars_cal[i]: {
                    "tar": float(mean_tpirs[i]),
                    "far": float(show_result["mean_fpir"][j]),
                    "frr": float(mean_fnirs[i]),
                    "threshold": float((g1_threshes[i] + g2_threshes[i]) / 2),
                }
                for j, i in enumerate(fars_show_idx)
            },
            fpir_targets=show_fars,
        )
        print(format_identification_table(mean_metrics, gallery_name="Mean (G1+G2)"))
        print(pd.DataFrame(show_result).set_index("fpir").to_markdown())
        return fars_cal, mean_tpirs, g1_cmc_scores, g2_cmc_scores, {
            "gallery1": g1_metrics,
            "gallery2": g2_metrics,
            "mean": mean_metrics,
        }


def load_pair_label(data_path, subset):
    backup_path = os.path.join(data_path, subset + "_backup.npz")
    if os.path.isfile(backup_path):
        _log_stage("Loading pair labels from cache: %s" % backup_path)
        label = np.load(backup_path)["label"]
        pos = int(np.sum(label == 1))
        neg = int(np.sum(label == 0))
        _log_stage("Pair labels ready: %d pairs (genuine=%d, impostor=%d)" % (len(label), pos, neg))
        return label

    pair_list_path = os.path.join(
        data_path, subset, "meta", "%s_template_pair_label.txt" % subset.lower()
    )
    if not os.path.isfile(pair_list_path):
        raise FileNotFoundError("Pair label file not found: %s" % pair_list_path)
    _log_stage("Loading pair labels from meta (slow, ~1–3 min): %s" % pair_list_path)
    _log_stage("Tip: run full eval once to create %s for faster replay" % backup_path)
    _, _, label = read_IJB_meta_columns_to_int(pair_list_path, columns=[0, 1, 2])
    pos = int(np.sum(label == 1))
    neg = int(np.sum(label == 0))
    _log_stage("Pair labels ready: %d pairs (genuine=%d, impostor=%d)" % (len(label), pos, neg))
    return label


def plot_roc_and_calculate_tpr(scores, names=None, label=None):
    _log_stage("Verification metrics (TAR/FRR/Threshold @ FAR) — starting")
    score_dict = {}
    for id, score in enumerate(scores):
        name = None if names is None else names[id]
        if isinstance(score, str) and score.endswith(".npz"):
            _log_stage("[1/5] Loading scores from %s ..." % score)
            aa = np.load(score)
            score = aa.get("scores", [])
            label = aa["label"] if label is None and "label" in aa else label
            score_name = aa.get("names", [])
            for ss, nn in zip(score, score_name):
                score_dict[nn] = np.asarray(ss, dtype=np.float64).ravel()
            _log_stage("[1/5] Scores loaded: %d method(s), %d pairs each (approx.)" % (
                len(score_dict), len(next(iter(score_dict.values()))) if score_dict else 0
            ))
        elif isinstance(score, str) and score.endswith(".npy"):
            name = name if name is not None else os.path.splitext(os.path.basename(score))[0]
            score_dict[name] = np.load(score).astype(np.float64).ravel()
        elif isinstance(score, str) and score.endswith(".txt"):
            label = pd.read_csv(score, sep=" ", header=None).values[:, 2]
        else:
            name = name if name is not None else str(id)
            score_dict[name] = np.asarray(score, dtype=np.float64).ravel()
    if label is None:
        print("Error: Label data is not provided")
        return None, None, None

    label = np.asarray(label, dtype=np.int8)
    if len(label) != len(next(iter(score_dict.values()))):
        raise ValueError(
            "Label count (%d) != score count (%d)" % (len(label), len(next(iter(score_dict.values()))))
        )

    pos_count = int(np.sum(label == 1))
    neg_count = int(np.sum(label == 0))
    _log_stage("[2/5] Labels aligned: %d pairs (genuine=%d, impostor=%d)" % (len(label), pos_count, neg_count))

    verification_reports = {}
    fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
    x_labels = [10 ** (-ii) for ii in range(1, 7)[::-1]]
    tpr_result = {}

    for step_idx, (name, score) in enumerate(score_dict.items(), start=3):
        _log_stage("[%d/5] Computing ROC on %d pairs for %s (typically 1–3 min) ..." % (
            step_idx, len(score), name
        ))
        fpr, tpr, thresholds = roc_curve(label, score)
        roc_auc = auc(fpr, tpr)
        fpr_dict[name], tpr_dict[name], roc_auc_dict[name] = fpr, tpr, roc_auc
        _log_stage("[%d/5] ROC done: %d curve points, AUC=%.6f — deriving TAR/FRR/EER ..." % (
            step_idx, len(fpr), roc_auc
        ))

        positives = score[label == 1]
        negatives = score[label == 0]
        verification_reports[name] = summarize_verification_metrics_from_roc(
            fpr,
            tpr,
            thresholds,
            pos_count,
            neg_count,
            far_targets=FAR_TARGETS_11_DISPLAY,
            auc=roc_auc,
            positive_scores=positives,
            negative_scores=negatives,
        )
        print(format_verification_table(verification_reports[name], method_name=name))

        fpr_flip, tpr_flip = np.flipud(fpr), np.flipud(tpr)
        tpr_result[name] = [float(tpr_flip[np.argmin(np.abs(fpr_flip - ii))]) for ii in x_labels]

    _log_stage("[5/5] Building summary tables ...")
    tpr_result_df = pd.DataFrame(tpr_result, index=x_labels).T
    tpr_result_df["AUC"] = pd.Series(roc_auc_dict)
    tpr_result_df.columns.name = "Methods"
    for name, report in verification_reports.items():
        tpr_result_df.loc[name, "EER"] = report.get("eer")
        for far_key, tar in report.get("frr_at_far", {}).items():
            tpr_result_df.loc[name, "FRR@%s" % far_key] = tar
        for far_key, thresh in report.get("threshold_at_far", {}).items():
            tpr_result_df.loc[name, "Th@%s" % far_key] = thresh
    try:
        print("\nLegacy TAR@FAR summary:\n" + tpr_result_df.to_markdown())
    except ImportError:
        print("\nLegacy TAR@FAR summary:\n" + tpr_result_df.to_string())

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        for name in score_dict:
            plt.plot(fpr_dict[name], tpr_dict[name], lw=1, label="[%s (AUC = %0.4f%%)]" % (name, roc_auc_dict[name] * 100))
        title = "ROC on IJB" + name.split("IJB")[-1][0] if "IJB" in name else "ROC on IJB"

        plt.xlim([10 ** -6, 0.1])
        plt.xscale("log")
        plt.xticks(x_labels)
        plt.xlabel("False Positive Rate")
        plt.ylim([0.3, 1.0])
        plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
        plt.ylabel("True Positive Rate")

        plt.grid(linestyle="--", linewidth=1)
        plt.title(title)
        plt.legend(loc="lower right", fontsize='x-small')
        plt.tight_layout()
        plt.show()
    except:
        print("matplotlib plot failed")
        fig = None

    _log_stage("Verification metrics complete.")
    return tpr_result_df, fig, verification_reports


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def save_metrics_json(metrics_payload, save_result_path):
    json_path = os.path.splitext(save_result_path)[0] + ".metrics.json"
    with open(json_path, "w", encoding="utf-8") as ff:
        json.dump(_json_safe(metrics_payload), ff, indent=2, ensure_ascii=False)
    print(">>>> Saved production metrics JSON:", json_path)
    return json_path


def plot_dir_far_cmc_scores(scores, names=None):
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        for id, score in enumerate(scores):
            name = None if names is None else names[id]
            if isinstance(score, str) and score.endswith(".npz"):
                aa = np.load(score)
                score, name = aa.get("scores")[0], aa.get("names")[0]
            fars, tpirs = score[0], score[1]
            name = name if name is not None else str(id)

            auc_value = auc(fars, tpirs)
            label = "[%s (AUC = %0.4f%%)]" % (name, auc_value * 100)
            plt.plot(fars, tpirs, lw=1, label=label)

        plt.xlabel("False Alarm Rate")
        plt.xlim([0.0001, 1])
        plt.xscale("log")
        plt.ylabel("Detection & Identification Rate (%)")
        plt.ylim([0, 1])

        plt.grid(linestyle="--", linewidth=1)
        plt.legend(fontsize='x-small')
        plt.tight_layout()
        plt.show()
    except:
        print("matplotlib plot failed")
        fig = None

    return fig


def parse_arguments(argv):
    import argparse

    default_save_result_name = "IJB_result/{model_name}_{subset}_{type}.npz"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model_file", type=str, default=None, help="Saved model, keras h5 / pytorch jit pth / onnx / mxnet")
    parser.add_argument("-d", "--data_path", type=str, default="./", help="Dataset path containing IJBB and IJBC sub folder")
    parser.add_argument("-s", "--subset", type=str, default="IJBC", help="Subset test target, could be IJBB / IJBC")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size for get_embeddings")
    parser.add_argument(
        "-R", "--save_result", type=str, default=default_save_result_name, help="Filename for saving / restore result"
    )
    parser.add_argument("-L", "--save_label", action="store_true", help="Save label data, useful for plot only")
    parser.add_argument("-E", "--save_embeddings", action="store_true", help="Save embeddings data")
    parser.add_argument("-B", "--is_bunch", action="store_true", help="Run all 8 tests N{0,1}D{0,1}F{0,1}")
    parser.add_argument("-N", "--is_one_2_N", action="store_true", help="Run 1:N test instead of 1:1")
    parser.add_argument("-F", "--force_reload", action="store_true", help="Force reload, instead of using cache")
    parser.add_argument("-P", "--plot_only", nargs="*", type=str, help="Plot saved results, Format 1 2 3 or 1, 2, 3 or *.npy")
    args = parser.parse_known_args(argv)[0]

    if args.plot_only != None and len(args.plot_only) != 0:
        # Plot only
        from glob2 import glob

        score_files = []
        for ss in args.plot_only:
            score_files.extend(glob(ss.replace(",", "").strip()))
        args.plot_only = score_files
    elif args.model_file == None and args.save_result == default_save_result_name:
        print("Please provide -m MODEL_FILE, see `--help` for usage.")
        exit(1)
    elif args.model_file != None:
        if args.model_file.endswith(".h5") or args.model_file.endswith(".pth") or args.model_file.endswith(".pt") or args.model_file.endswith(".onnx"):
            # Keras model file "model.h5", pytorch model ends with `.pth` or `.pt`, onnx model ends with `.onnx`
            model_name = os.path.splitext(os.path.basename(args.model_file))[0]
        else:
            # MXNet model file "models/r50-arcface-emore/model,1"
            model_name = os.path.basename(os.path.dirname(args.model_file))

        if args.save_result == default_save_result_name:
            type = "1N" if args.is_one_2_N else "11"
            args.save_result = default_save_result_name.format(model_name=model_name, subset=args.subset, type=type)
    return args


if __name__ == "__main__":
    import sys

    args = parse_arguments(sys.argv[1:])
    if args.plot_only != None and len(args.plot_only) != 0:
        if args.is_one_2_N:
            plot_dir_far_cmc_scores(args.plot_only)
        else:
            label = None
            pair_label_path = os.path.join(
                args.data_path, args.subset, "meta",
                "%s_template_pair_label.txt" % args.subset.lower(),
            )
            backup_path = os.path.join(args.data_path, args.subset + "_backup.npz")
            if os.path.isfile(backup_path) or os.path.isfile(pair_label_path):
                label = load_pair_label(args.data_path, args.subset)
            _, _, verification_reports = plot_roc_and_calculate_tpr(args.plot_only, label=label)
            if verification_reports:
                metrics_payload = {
                    "subset": args.subset,
                    "protocol": "N0D1F1",
                    "task": "1:1",
                    "plot_only": True,
                    "verification": verification_reports,
                }
                for plot_file in args.plot_only:
                    if plot_file.endswith(".npz"):
                        save_metrics_json(metrics_payload, plot_file)
                        break
    else:
        save_name = os.path.splitext(os.path.basename(args.save_result))[0]
        save_items = {}
        save_path = os.path.dirname(args.save_result)
        if len(save_path) != 0 and not os.path.exists(save_path):
            os.makedirs(save_path)

        tt = IJB_test(args.model_file, args.data_path, args.subset, args.batch_size, args.force_reload, args.save_result)
        if args.save_embeddings:  # Save embeddings first, in case of any error happens later...
            np.savez(args.save_result, embs=tt.embs, embs_f=tt.embs_f)

        metrics_payload = {
            "subset": args.subset,
            "protocol": "N0D1F1",
            "result_file": args.save_result,
        }

        if args.is_one_2_N:  # 1:N test
            fars, tpirs, _, _, id_metrics = tt.run_model_test_1N()
            scores = [(fars, tpirs)]
            names = [save_name]
            save_items.update({"scores": scores, "names": names})
            metrics_payload.update({"task": "1:N", "identification": id_metrics})
        elif args.is_bunch:  # All 8 tests N{0,1}D{0,1}F{0,1}
            scores, names = tt.run_model_test_bunch()
            names = [save_name + "_" + ii for ii in names]
            label = tt.label
            save_items.update({"scores": scores, "names": names})
            metrics_payload.update({"task": "1:1_bunch"})
        else:  # Basic 1:1 N0D1F1 test
            score = tt.run_model_test_single()
            scores, names, label = [score], [save_name], tt.label
            save_items.update({"scores": scores, "names": names})
            metrics_payload.update({"task": "1:1"})

        if args.save_embeddings:
            save_items.update({"embs": tt.embs, "embs_f": tt.embs_f})
        if args.save_label:
            save_items.update({"label": label})

        if args.model_file != None or args.save_embeddings:
            np.savez(args.save_result, **save_items)

        if args.is_one_2_N:
            plot_dir_far_cmc_scores(scores=scores, names=names)
            save_metrics_json(metrics_payload, args.save_result)
        else:
            _, _, verification_reports = plot_roc_and_calculate_tpr(scores, names=names, label=label)
            if verification_reports:
                metrics_payload["verification"] = verification_reports
            save_metrics_json(metrics_payload, args.save_result)
