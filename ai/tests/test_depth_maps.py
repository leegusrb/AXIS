import json

import numpy as np
from PIL import Image

from axis_ai.data.depth_maps import load_depth_maps, make_depth_maps


class FakeDepth:
    calls = 0

    def estimate(self, img):
        FakeDepth.calls += 1
        return np.full((4, 3), img.getpixel((0, 0))[0] / 255, np.float32)


def test_make_depth_maps_skips_existing(tmp_path):
    proc, raw = tmp_path / "proc", tmp_path / "raw"
    proc.mkdir(), raw.mkdir()
    with open(proc / "features.jsonl", "w") as f:
        for i, v in enumerate((51, 102)):
            Image.new("RGB", (30, 40), (v, 0, 0)).save(raw / f"{i}.jpg")
            f.write(json.dumps({"image_id": f"id{i}", "file": f"{i}.jpg"}) + "\n")
    assert make_depth_maps(proc, raw, FakeDepth()) == 2
    maps = load_depth_maps(proc)
    assert set(maps) == {"id0", "id1"} and maps["id1"].dtype == np.float16
    assert abs(float(maps["id0"][0, 0]) - 0.2) < 0.01
    FakeDepth.calls = 0
    assert make_depth_maps(proc, raw, FakeDepth()) == 0 and FakeDepth.calls == 0
