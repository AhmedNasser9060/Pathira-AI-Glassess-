import io

import numpy as np
from PIL import Image

from backend.ml import obstacle


class _Tensor:
    def __init__(self, value):
        self._value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self._value


class _Boxes:
    def __init__(self, rows):
        self.xyxy = _Tensor([row[:4] for row in rows])
        self.conf = _Tensor([row[4] for row in rows])
        self.cls = _Tensor([row[5] for row in rows])

    def __len__(self):
        return len(self.conf.numpy())


class _Result:
    def __init__(self, names, rows):
        self.names = names
        self.boxes = _Boxes(rows)


class _Model:
    def __init__(self, names, rows):
        self.names = names
        self._rows = rows
        self.calls = 0

    def predict(self, image, conf, verbose):
        self.calls += 1
        assert image.size == (640, 480)
        assert conf == 0.25
        assert verbose is False
        return [_Result(self.names, self._rows)]


def _image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (640, 480), "white").save(output, format="PNG")
    return output.getvalue()


def test_dual_models_merge_and_normalize(monkeypatch):
    old = _Model(
        {0: "wall", 1: "door", 2: "stairs", 3: "ignored"},
        [[0, 0, 200, 300, 0.9, 0], [10, 10, 20, 20, 0.99, 3]],
    )
    new = _Model(
        {0: "Bench", 1: "Person", 2: "Chair", 3: "Table"},
        [[300, 50, 500, 470, 0.95, 1], [220, 100, 360, 300, 0.8, 3]],
    )
    monkeypatch.setattr(obstacle, "get_obstacle_yolo", lambda: old)
    monkeypatch.setattr(obstacle, "get_objects_yolo", lambda: new)

    result = obstacle.run_obstacle_sync(_image_bytes())

    assert old.calls == 1 and new.calls == 1
    assert {item["label"] for item in result["detections"]} == {"wall", "Person", "Table"}
    assert all(item["is_critical"] is True for item in result["detections"])
    assert result["highest_priority"] == result["detections"][0]
    assert result["voice_guidance"]


def test_dual_models_clear_path(monkeypatch):
    old = _Model({0: "wall", 1: "door", 2: "stairs"}, [])
    new = _Model({0: "Bench", 1: "Person", 2: "Chair", 3: "Table"}, [])
    monkeypatch.setattr(obstacle, "get_obstacle_yolo", lambda: old)
    monkeypatch.setattr(obstacle, "get_objects_yolo", lambda: new)

    result = obstacle.run_obstacle_sync(_image_bytes())

    assert result["detections"] == []
    assert result["highest_priority"] is None
    assert result["voice_guidance"] == ["Path is clear"]
