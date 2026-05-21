import pytest
import os


def test_main_window_smoke(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    PySide6 = pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QLabel, QPushButton

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
    cfg = AppConfig(workspace_path=str(tmp_path), auto_load_model=False, safe_mode=True, recognition_threshold=0.61)
    storage = Storage(cfg.database_path)
    engine = FaceEngine(model_name=cfg.model_name)
    window = MainWindow(StudioContext(cfg, True, storage, engine, str(tmp_path / "app.log")))
    window.show()
    assert window.mode_combo.minimumWidth() >= 250
    window.open_page("verification")
    verification_page = window.page_registry.get("verification")
    assert abs(window.context.config.recognition_threshold - 0.28) < 1e-9
    assert abs(verification_page.threshold.value() - 0.28) < 1e-9
    assert hasattr(verification_page.result_table, "_proportional_table_sizer")
    label_texts = [label.text() for label in verification_page.findChildren(QLabel)]
    assert "Mode: waiting for gallery" not in label_texts
    for button in verification_page.findChildren(QPushButton):
        assert button.toolTip()
    face_swap_page = window.page_registry.get("image_face_swap")
    assert face_swap_page.output_view.objectName() == "imageViewer"
    assert face_swap_page.output_view.viewport().objectName() == "imageViewerViewport"
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
    assert not hasattr(model_dialog.runtime_page, "threshold")
    assert not hasattr(model_dialog.runtime_page, "workers")
    assert not hasattr(model_dialog.runtime_page, "frame_interval")
    assert hasattr(model_dialog.runtime_page, "gfpgan_enabled")
    assert hasattr(model_dialog.runtime_page, "gfpgan_model_combo")
    dialogs = [settings_dialog, model_dialog, LicenseDialog(window.context, window)]
    for dialog in dialogs:
        dialog.close()
    window.close()
    assert window.windowTitle().startswith("InsightFace Evaluation Studio")
