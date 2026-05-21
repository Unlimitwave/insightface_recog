import numpy as np

from insightface.gui.core.clustering import (
    HDBSCANUnavailableError,
    cluster_embeddings_dbscan,
    cluster_embeddings_hdbscan_auto,
    hdbscan_status,
)


def test_dbscan_default_distance_threshold_groups_near_faces():
    embeddings = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.98, 0.08], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([0.08, 0.98], dtype=np.float32),
    ]

    labels, algorithm = cluster_embeddings_dbscan(embeddings, min_samples=2)

    assert algorithm in {"DBSCAN", "centroid fallback"}
    assert len(labels) == 4
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_hdbscan_auto_returns_labels_without_threshold_input():
    embeddings = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.98, 0.08], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([0.08, 0.98], dtype=np.float32),
    ]

    labels, algorithm = cluster_embeddings_hdbscan_auto(
        embeddings,
        min_cluster_size=2,
        min_samples=1,
    )

    assert "HDBSCAN" in algorithm
    assert len(labels) == 4
    assert all(isinstance(label, int) for label in labels)


def test_hdbscan_status_controls_album_availability():
    available, message = hdbscan_status()
    assert isinstance(available, bool)
    assert isinstance(message, str)
    assert message


def test_hdbscan_does_not_fallback_when_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sklearn.cluster" or name == "hdbscan":
            raise ImportError("missing hdbscan for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        cluster_embeddings_hdbscan_auto(
            [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)],
            min_cluster_size=2,
        )
    except HDBSCANUnavailableError as exc:
        assert "HDBSCAN is required" in str(exc)
    else:
        raise AssertionError("Expected HDBSCANUnavailableError")
