
Download megaface testsuite from [baiducloud](https://pan.baidu.com/s/1Vdxc2GgbY8wIW0hVcObIwg)(code:0n6w) or [gdrive](https://drive.google.com/file/d/1KBwp0U9oZgZj7SYDXRxUnnH7Lwvd9XMy/view?usp=sharing). The official devkit is also included.

## Buffalo_l ONNX Evaluation

Use `buffalo_l` (or any InsightFace ONNX recognition model) without MXNet checkpoints.

### 1. Prepare dataset images

```bash
cd /path/to/megaface/data
rm -f facescrub_images megaface_images
unzip megaface_testpack_v1.0.zip
```

### 2. One-command evaluation

```bash
cd recognition/_evaluation_/megaface
chmod +x run_buffalo_l.sh

# default paths:
#   dataset: /home/cmsr/桌面/东风/数据集/megaface
#   model:   ~/.insightface/models/buffalo_l/w600k_r50.onnx
./run_buffalo_l.sh
```

Environment overrides:

```bash
MEGAFACE_ROOT=/path/to/megaface \
MODEL_FILE=~/.insightface/models/buffalo_l/w600k_r50.onnx \
GPU=0 \
BATCH_SIZE=32 \
FORCE_REEXTRACT=1 \
./run_buffalo_l.sh
```

`run_buffalo_l.sh` auto-configures conda `nvidia-*` CUDA libraries via `LD_LIBRARY_PATH`.
Use `FORCE_REEXTRACT=1` to rebuild features instead of resuming from existing `.bin` files.

Quick smoke test on a smaller gallery:

```bash
GALLERY_SIZE=100 ./run_buffalo_l.sh
```

### 3. Manual steps

```bash
python gen_megaface_onnx.py \
  --model-file ~/.insightface/models/buffalo_l/w600k_r50.onnx \
  --facescrub-root /path/to/data/facescrub_images \
  --megaface-root /path/to/data/megaface_images \
  --facescrub-lst /path/to/data/facescrub_lst \
  --megaface-lst /path/to/data/megaface_lst \
  --output ./feature_out/buffalo_l \
  --skip-existing

python remove_noises.py --algo buffalo_l \
  --feature-dir-input ./feature_out/buffalo_l \
  --feature-dir-out ./feature_out_clean/buffalo_l \
  --facescrub-lst /path/to/data/facescrub_lst \
  --megaface-lst /path/to/data/megaface_lst

python run_experiment_py3.py \
  --devkit-root /path/to/megaface/devkit \
  ./feature_out_clean/buffalo_l/megaface \
  ./feature_out_clean/buffalo_l/facescrub \
  _buffalo_l.bin ./results/buffalo_l -s 1000000

python parse_megaface_results.py \
  --result-dir ./results/buffalo_l --algo buffalo_l
```

### Metrics

| Output | Meaning |
|--------|---------|
| `MegaFace Rank-1 (Id)` | 1:N identification with gallery=1M |
| `Verification @ FAR=1e-6 (Ver)` | 1:1 verification on MegaFace |

Model Zoo `buffalo_l` MegaFace column corresponds to **Rank-1 @ gallery=1e6**.

### Notes

- Feature extraction supports `--skip-existing` for resume.
- Final scoring still uses the official MegaFace devkit binaries (`Identification`, `FuseResults`).
- If devkit binaries fail with missing `libopencv_core.so.2.4`, install OpenCV 2.4 runtime libs or provide them via `LD_LIBRARY_PATH`.

### Legacy MXNet pipeline

The original `gen_megaface.py` + `run.sh` path remains available for old MXNet checkpoints.
