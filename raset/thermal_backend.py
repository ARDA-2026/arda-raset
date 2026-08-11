"""threshold(thermal_main.py) / YOLO(thermal_main_yolo.py) 판정 백엔드를
공용 인터페이스로 감싸는 어댑터.

두 스크립트의 run_observation() 상태기계(dwell/settle/give-up 카운트다운/
선점, ~120줄)는 완전히 동일하고 판정 로직(~10줄)만 다르다 — 그 상태기계를
thermal_worker.py에 한 번만 작성하기 위해 판정 부분만 여기서 분리한다.

thermal-camera/src가 sys.path에 있어야 import된다(main.py에서 삽입).
offset_from_circle_x/y·WIDTH/HEIGHT 등은 두 백엔드 모두 값이 완전히 같아서
(같은 32x24 MLX90640 그리드 기준) thermal_main 쪽 것만 공용으로 쓴다.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

import thermal_main

WIDTH = thermal_main.WIDTH
HEIGHT = thermal_main.HEIGHT
FRAME_INTERVAL_S = thermal_main.FRAME_INTERVAL_S
SENSOR_WARMUP_FRAMES = thermal_main.SENSOR_WARMUP_FRAMES

initialize_sensor = thermal_main.initialize_sensor
read_frame = thermal_main.read_frame
create_absolute_colormap = thermal_main.create_absolute_colormap
offset_from_circle_x = thermal_main.offset_from_circle_x
offset_from_circle_y = thermal_main.offset_from_circle_y


@dataclass
class Detection:
    matched: bool
    grid_xy: tuple[float, float] | None
    state: Any  # draw()에 그대로 넘길 백엔드별 상세 정보 (matched=True면 grid_xy도 항상 있음)


class Backend:
    """threshold/YOLO 공용 인터페이스. detect()는 매 프레임 1회, draw()는
    confirmed 여부가 정해진 뒤(호출 시점 인자로) 1회 호출된다."""

    def detect(self, thermal: np.ndarray, color_image: np.ndarray) -> Detection:
        raise NotImplementedError

    def draw(self, image: np.ndarray, detection: Detection, confirmed: bool) -> None:
        raise NotImplementedError


class ThresholdBackend(Backend):
    """온도 임계값 + 원형도 기반 판정 (thermal_main.py, 기본 백엔드)."""

    def detect(self, thermal, color_image):
        detection, _bg, _thr = thermal_main.detect_hot_region(thermal)
        matched = thermal_main.is_head_shape(detection)
        grid_xy = detection["circle"][:2] if detection else None
        return Detection(matched=matched, grid_xy=grid_xy, state=detection)

    def draw(self, image, detection, confirmed):
        thermal_main.draw_detection(image, detection.state, detection.matched, confirmed)


class YoloBackend(Backend):
    """커스텀 YOLO 모델 기반 판정 (thermal_main_yolo.py, --yolo)."""

    def __init__(self, model_path: str, device: str, confidence_threshold: float):
        import thermal_main_yolo

        self._mod = thermal_main_yolo
        self.model, self.device = thermal_main_yolo.load_yolo_model(model_path, device)
        self.confidence_threshold = confidence_threshold

    def detect(self, thermal, color_image):
        mod = self._mod
        candidates = mod.detect_person_candidates(
            color_image, self.model, self.device, self.confidence_threshold,
        )
        best = mod.best_candidate(candidates)
        grid_xy = mod.candidate_to_grid_xy(best) if best else None
        return Detection(matched=best is not None, grid_xy=grid_xy, state=(candidates, best))

    def draw(self, image, detection, confirmed):
        candidates, best = detection.state
        self._mod.draw_person_candidates(image, candidates, best, confirmed)
