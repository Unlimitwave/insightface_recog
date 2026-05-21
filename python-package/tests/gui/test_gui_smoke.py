import pytest
import os
from pathlib import Path


def test_main_window_smoke(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    PySide6 = pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QAbstractItemView, QApplication, QLabel, QPushButton

    from insightface.gui.app import StudioContext, configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.face_engine import FaceEngine
    from insightface.gui.core.navigation import AppMode
    from insightface.gui.core.storage import Storage
    from insightface.gui.dialogs.license_dialog import LicenseDialog
    from insightface.gui.dialogs.model_manager_dialog import ModelManagerDialog
    from insightface.gui.dialogs.settings_dialog import SettingsDialog
    from insightface.gui.main_window import MainWindow

    configure_qt_plugin_paths()
    app = QApplication.instance() or QApplication([])
    cfg = AppConfig(
        workspace_path=str(tmp_path),
        model_root=str(tmp_path / "model-root"),
        auto_load_model=False,
        safe_mode=True,
        recognition_threshold=0.61,
    )
    storage = Storage(cfg.database_path)
    engine = FaceEngine(model_name=cfg.model_name)
    window = MainWindow(StudioContext(cfg, True, storage, engine, str(tmp_path / "app.log")))
    window.show()
    assert window.mode_combo.minimumWidth() >= 250
    window.open_page("verification")
    verification_page = window.page_registry.get("verification")
    assert abs(window.context.config.recognition_threshold - 0.28) < 1e-9
    assert abs(verification_page.threshold.value() - 0.28) < 1e-9
    notice = verification_page.findChild(QLabel, "noticeLabel")
    assert notice is not None
    assert "Gallery face embeddings are cached in memory" in notice.text()
    assert "All processing is local by default" not in notice.text()
    assert hasattr(verification_page.result_table, "_proportional_table_sizer")
    assert verification_page.result_table.selectionBehavior() == QAbstractItemView.SelectRows
    assert verification_page.result_table.selectionMode() == QAbstractItemView.SingleSelection
    verification_page._gallery_embedding_cache_key = ("old.jpg",)
    verification_page._gallery_embedding_cache = [{"path": "old.jpg"}]
    verification_page.set_gallery_paths(["new.jpg"])
    assert verification_page._gallery_embedding_cache_key is None
    assert verification_page._gallery_embedding_cache is None
    label_texts = [label.text() for label in verification_page.findChildren(QLabel)]
    assert "Mode: waiting for gallery" not in label_texts
    for button in verification_page.findChildren(QPushButton):
        assert button.toolTip()
    face_swap_page = window.page_registry.get("image_face_swap")
    assert face_swap_page.output_view.objectName() == "imageViewer"
    assert face_swap_page.output_view.viewport().objectName() == "imageViewerViewport"
    album_page = window.page_registry.get("album")
    assert not hasattr(album_page, "cluster_threshold")
    assert not hasattr(album_page, "match_threshold")
    assert album_page.min_face_size.value() == 80
    assert album_page.algorithm_label.text().startswith("Algorithm: HDBSCAN")
    assert hasattr(album_page, "import_button")
    assert hasattr(album_page, "rebuild_button")
    assert album_page.cluster_table.columnCount() == 2
    assert album_page.cluster_table.horizontalHeaderItem(0).text() == "Thumbnail"
    assert album_page.cluster_table.horizontalHeaderItem(1).text() == "Photos"
    assert album_page.cluster_table.selectionBehavior() == QAbstractItemView.SelectRows
    assert album_page.photo_table.selectionBehavior() == QAbstractItemView.SelectRows
    for mode in AppMode:
        window.change_mode(mode)
        assert window.sidebar_list.count() > 0
        if mode in {AppMode.FACE_VERIFICATION, AppMode.ALBUM_MANAGEMENT, AppMode.FACE_SWAP}:
            assert not window.sidebar.isVisible()
        else:
            assert window.sidebar.isVisible()
        sidebar_titles = [window.sidebar_list.item(i).text() for i in range(window.sidebar_list.count())]
        assert "Settings" not in sidebar_titles
        assert "Model Settings" not in sidebar_titles
        assert "Model Downloads" not in sidebar_titles
        assert "License Center" not in sidebar_titles
    settings_dialog = SettingsDialog(window.context, window)
    assert hasattr(settings_dialog, "theme")
    assert settings_dialog.theme.count() >= 7
    assert not hasattr(settings_dialog, "workspace")
    assert not hasattr(settings_dialog, "default_mode")
    model_dialog = ModelManagerDialog(window.context, window)
    assert model_dialog.minimumWidth() >= 1120
    assert model_dialog.downloads_page.table.minimumHeight() >= 400
    assert hasattr(model_dialog, "run_task")
    assert model_dialog.downloads_page.table.selectionBehavior() == QAbstractItemView.SelectRows
    assert model_dialog.downloads_page.table.selectionMode() == QAbstractItemView.SingleSelection
    assert not hasattr(model_dialog.runtime_page, "threshold")
    assert not hasattr(model_dialog.runtime_page, "workers")
    assert not hasattr(model_dialog.runtime_page, "frame_interval")
    assert hasattr(model_dialog.runtime_page, "gfpgan_enabled")
    assert hasattr(model_dialog.runtime_page, "gfpgan_model_combo")
    assert not model_dialog.runtime_page.gfpgan_enabled.isEnabled()
    gfpgan_path = Path(cfg.model_root) / "models" / "GFPGANv1.4" / "GFPGANv1.4.onnx"
    gfpgan_path.parent.mkdir(parents=True)
    gfpgan_path.write_bytes(b"fake")
    model_dialog.runtime_page.refresh()
    assert model_dialog.runtime_page.gfpgan_enabled.isEnabled()
    dialogs = [settings_dialog, model_dialog, LicenseDialog(window.context, window)]
    for dialog in dialogs:
        dialog.close()
    window.close()
    assert window.windowTitle().startswith("InsightFace Evaluation Studio")
