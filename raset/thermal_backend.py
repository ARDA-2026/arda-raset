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
import thermal_main_yolo

WIDTH = thermal_main.WIDTH
HEIGHT = thermal_main.HEIGHT
FRAME_INTERVAL_S = thermal_main.FRAME_INTERVAL_S
SENSOR_WARMUP_FRAMES = thermal_main.SENSOR_WARMUP_FRAMES

initialize_sensor = thermal_main.initialize_sensor
read_frame = thermal_main.read_frame
create_absolute_colormap = thermal_main.create_absolute_colormap
offset_from_circle_x = thermal_main.offset_from_circle_x
offset_from_circle_y = thermal_main.offset_from_circle_y

# 배경-상대 보정 컬러맵(YOLO 입력 전용, thermal_main_yolo.py 참고) — YOLO
# 모델 로드(무거움, 지연 import)와 달리 이 함수는 순수 numpy/cv2 연산이라
# 백엔드 종류(threshold/YOLO)와 무관하게 항상 가져다 쓸 수 있다. 웹
# 스트리밍에서 "욜로가 보는 이미지"를 고를 때(thermal_worker.py의
# stream_view 옵션) threshold 백엔드로 실행 중이어도 이 미리보기를 그대로
# 쓸 수 있게 하기 위함.
create_relative_colormap = thermal_main_yolo.create_relative_colormap


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
        # 화면 표시/스트리밍용 color_image(실제 온도 그대로, 절대 범위)는 안
        # 쓰고, 배경(중앙값) 기준 상대 보정 컬러맵을 따로 만들어 YOLO에
        # 넣는다 — 배경 온도가 바뀌어도 "사람-배경 온도차"가 학습 때와
        # 비슷한 색 대비로 나오게 하기 위함(mod.create_relative_colormap
        # 참고 — thermal_main_yolo.py에 구현돼 있음).
        yolo_input = mod.create_relative_colormap(thermal)
        candidates = mod.detect_person_candidates(
            yolo_input, self.model, self.device, self.confidence_threshold,
        )
        best = mod.best_candidate(candidates)
        grid_xy = mod.candidate_to_grid_xy(best) if best else None
        return Detection(matched=best is not None, grid_xy=grid_xy, state=(candidates, best))

    def draw(self, image, detection, confirmed):
        candidates, best = detection.state
        self._mod.draw_person_candidates(image, candidates, best, confirmed)


def detection_detail(detection: Detection) -> dict:
    """detection.state(백엔드별로 구조가 다름)에서 "사람인지" 판정에 실제로
    쓰이는 수치를 뽑아 공용 dict로 만든다 — 로그/진단 전용, 판정 로직 자체는
    읽기만 하고 손대지 않는다.

    ThresholdBackend(기본)는 원형도(circularity)/종횡비/채움비율/코어중심
    오프셋 — 전부 thermal_main.is_head_shape()가 실제로 비교하는 값과
    그 임계값을 같이 넣어서, 지금 값이 임계값에 얼마나 가까운지("사람인지
    잡는 그 포인트") 바로 볼 수 있게 한다. YoloBackend(--yolo)는 최고
    확률 후보의 confidence만 있다."""
    state = detection.state
    if isinstance(state, dict):  # ThresholdBackend — thermal_main.detect_hot_region() 반환값
        return {
            "backend": "threshold",
            "circularity": state.get("circularity"),
            "circularity_min": thermal_main.MIN_CIRCULARITY,
            "aspect_ratio": state.get("aspect_ratio"),
            "aspect_ratio_range": [thermal_main.MIN_ASPECT_RATIO, thermal_main.MAX_ASPECT_RATIO],
            "fill_ratio": state.get("fill_ratio"),
            "fill_ratio_min": thermal_main.MIN_FILL_RATIO,
            "core_center_offset_ratio": state.get("core_center_offset_ratio"),
            "core_center_offset_ratio_max": thermal_main.MAX_CORE_CENTER_OFFSET_RATIO,
            "pixel_area": state.get("pixel_area"),
        }
    if isinstance(state, tuple) and len(state) == 2:  # YoloBackend — (candidates, best)
        _candidates, best = state
        return {
            "backend": "yolo",
            "confidence": best.get("confidence") if best else None,
        }
    return {"backend": "none"}  # 이번 프레임은 발열 영역 자체를 못 찾음(state=None)
