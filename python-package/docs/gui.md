# InsightFace Evaluation Studio GUI

InsightFace Evaluation Studio is a local desktop GUI for InsightFace 1.0. It is
designed for no-code face recognition testing, local People Library management,
album organization, enterprise model evaluation, report export, and basic face
swap trials.

## Install

```bash
cd python-package
pip install -e ".[gui]"
insightface-gui
```

PyPI users can install:

```bash
pip install "insightface[gui]"
insightface-gui
```

Aliases:

```bash
insightface-eval-studio
insightface-desktop
python -m insightface.gui
```

## Workspace

By default, user data is stored in:

```text
~/.insightface/gui
```

The workspace contains the SQLite database, crops, exports, reports, and logs.
The GUI does not upload images, videos, embeddings, or reports automatically.

You can override the workspace:

```bash
insightface-gui --workspace /path/to/workspace
```

## File, Folder, and Video Inputs

Image and video pages use the preview frame as the upload target. Click the
empty preview frame labeled `Click to upload or drag a file here`, or drag a
local file onto it. The preview changes color on hover to indicate it is
clickable, and changes again while a valid file is dragged over it. After a file
is loaded, the small `×` button removes it, and dragging another file onto the
preview replaces it.

Multi-file imports, folder imports, CSV files, and local model inputs use
clickable drag-and-drop selectors with hover and drag-over color feedback.

## Mode-based navigation

InsightFace Evaluation Studio uses four workflow modes. The mode selector is in
the top app bar. Face Recognition, Album Management, and Face Swap use a single
full-width workspace, while modes with several workflows show a compact left
sidebar:

1. **Face Recognition**: one Query & Gallery workspace. One gallery image runs
   1:1 compare; multiple gallery images or a folder run 1:N gallery search.
2. **Album Management**: one **Album** workspace for local folder import,
   refresh, DBSCAN face clustering, and photo review.
3. **Face Swap**: one Source + Target = Result workspace. Target can be an
   image or video.
4. **Enterprise Evaluation**: local business evaluation, dataset setup,
   threshold calibration, reports, and commercial next steps.

Global utilities are always available from the top app bar:

- **Settings** opens the application settings dialog for the UI theme. The
  included themes are System, Precision Light, Studio Dark, Graphite Pro, Azure
  Lab, Emerald Focus, and Crimson Audit. Workspace paths are chosen on first
  launch and are not changed from this dialog.
- **Models** opens runtime settings, model downloads, and custom model directory
  tools.
- **License** opens the License Center dialog.

Settings, Model Settings, Model Downloads, and License Center are intentionally
not shown in the left sidebar.

## Models

Open **Models** from the top app bar or **Tools > Models** to choose:

- model pack: `buffalo_l`, `buffalo_s`, `antelopev2`, or a custom model folder
- provider: Auto, CPU, CUDA when `CUDAExecutionProvider` is available
- detection size: Auto, 128x128, 320x320, 640x640, 1024x1024
- face swap model
- GFPGAN post-processing model, if `GFPGANv1.4.onnx` has been downloaded

**Auto** detection size is the default. It runs joint 128x128 and 640x640
detection and merges duplicate boxes.

The Runtime tab also lets you choose a face swap model. Only downloaded local
swap models are listed. Download `inswapper_128.onnx` or another compatible
swap model from **Models > Downloads** first.

The Runtime tab can also enable GFPGAN post-processing for face swap. GFPGAN is
listed in **Models > Downloads** as a third-party restore model and is not
downloaded automatically. When enabled, face swap output is passed through the
configured GFPGAN ONNX model using a 512x512 restore pass.

The GUI opens even when a model is missing. In that case, pages show
`Model is not loaded. Please open Models.`

The GUI does not download models automatically. Open **Models > Downloads** and
click **Refresh Download URLs** to fetch the latest release assets from:

```text
https://github.com/deepinsight/insightface/releases
```

The refreshed URLs are cached in:

```text
~/.insightface/gui/cache/model_download_urls.json
```

The GFPGANv1.4 third-party entry is added by the GUI alongside refreshed
InsightFace release assets. It points to the Hugging Face `resolve/main` file
URL for `Gourieff/ReActor` and is saved under the same local model root after
manual download.

Downloaded archives are cached in:

```text
~/.insightface/gui/cache/models
```

Zip model packages are extracted to:

```text
~/.insightface/models/<model_name>/
```

For example:

```text
~/.insightface/models/buffalo_l/
~/.insightface/models/antelopev2/
```

Users can also manually place model directories under `~/.insightface/models/`
or choose **custom model directory** from **Models > Runtime** to reveal the
custom directory field.

## Face Recognition

Open **Face Recognition** and use the full-width **Query & Gallery** workspace.
The Query preview accepts one image by click or drag. The Gallery panel accepts
one image, multiple images, or a folder by click or drag. There is no separate
Choose button in Gallery; click the Gallery panel or drag files/folders onto it.

If Gallery contains one image, **Run Recognition** automatically runs 1:1
compare and reports similarity, threshold, decision, and detection score. If
Gallery contains multiple images or a folder, the same button runs 1:N gallery
search and ranks the gallery images by similarity.

## Multi-face Photo Recognition

Open **Multi-face Photo Recognition** to detect all faces in a group photo,
identify them against People Library, save results to the local database, and
export annotated images or CSV/JSON.

## Batch Processing

Open **Batch Folder Processing**, select an image folder, and choose recursive
scan, crop saving, and identification options. Batch results can be exported to
CSV and JSON.

## Album People Clustering

Open **Album Management > Album** to add one or more album directories. Click
**Import / Refresh** to scan new image files, detect faces, save local crops,
and cluster all indexed faces from the selected directories. The page uses
DBSCAN with a default cosine-distance threshold of `0.28`. If scikit-learn is
not available, it falls back to a simple centroid grouping strategy and shows
the algorithm used.

Album cluster IDs avoid duplicating existing People Library IDs. When a cluster
matches an existing person within the configured duplicate distance threshold
(`0.28` by default), the existing person ID is reused; otherwise the page assigns
the next available album person ID. The cluster thumbnail is chosen from the
face nearest the cluster centroid. Selecting a cluster shows all original photo
thumbnails for that cluster, and double-clicking a thumbnail opens the original
image.

Album directories and clustering results are saved in the local SQLite
database so the page can restore them on the next launch. **Clear** only clears
the selected album directories and leaves the current clustering results
visible. **Rebuild All** asks for confirmation, then reprocesses all selected
album directories from scratch and replaces the saved clustering results. If no
album directories are selected, **Rebuild All** clears the saved clustering
results.

## Enterprise Evaluation

Open **Enterprise Evaluation** to run no-code local evaluations:

- KYC / 1:1 Verification with a pairs CSV
- Access Control / 1:N Identification with gallery and probe folders

Other scenarios are represented as guided placeholders in v1.0 and can be
handled through their dedicated pages.

## Report Export

Reports are written to:

```text
~/.insightface/gui/reports
```

Markdown, HTML, and PDF are supported when the GUI extra is installed.

## Face Swap

Open **Face Swap** to use the full-width Source + Target = Result workspace.
Choose the swap model in **Models > Runtime**; the page itself does not expose a
model picker. The swap model is loaded only when you click **Run Face Swap**.
If GFPGAN post-processing is enabled in **Models > Runtime**, the swapped image
or swapped video frames are restored with GFPGAN before being saved.

Source is always an image. Target can be an image or a video, and the workflow
automatically chooses image swap or video swap from the target file type. Video
swap writes an `.mp4` result to the exports folder and shows a preview frame
when one is available.

If the model is missing, it shows:

```text
Face swap model not found. Please download and choose a swap model in Models.
```

Face swap may require separate commercial authorization depending on usage and
model license. Use only with appropriate rights and consent.

## License Notice

Code and model files may have different licenses. Research or publicly
distributed pretrained models may be restricted to non-commercial or research
usage. Commercial deployment requires appropriate model authorization.

This tool does not provide legal advice. Users are responsible for consent,
privacy, retention, and compliance with applicable biometric regulations.

## Troubleshooting

Run without automatic model loading:

```bash
insightface-gui --safe-mode
```

Safe mode is only applied to the current launch. It is intended for
troubleshooting model/provider issues and is not saved to `config.json`.

Force CPU:

```bash
insightface-gui --provider cpu
```

`CUDA` is selectable only when ONNX Runtime reports `CUDAExecutionProvider`.
If CUDA is requested on a machine without a usable CUDA provider, the GUI
falls back to Auto/CPU instead of exposing a broken GPU option.

Logs are stored in:

```text
~/.insightface/gui/logs/app.log
```
