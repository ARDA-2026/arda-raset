"""열화상(thermal-camera) 서브시스템 구동 — threshold/YOLO 공용.

thermal_main.py / thermal_main_yolo.py의 run_observation() 상태기계(대기
트리거 폴링 → 프레임 읽기 → 판정 → 서보 추적보정 전송 → settle 여부로
give-up 카운트다운 시작/보류 → 연속매칭 카운트 → 확정/포기 → 회신)를 그대로
이식하되, UDP 대신 bus 큐를 쓴다.

기본적으로 cv2.imshow(GUI 창)는 안 쓴다 — 대신 관찰(dwell) 중인 동안 매
프레임(~0.5초 간격) 기존 lat/lon/timestamp 리포트 포맷
(`arda.utils.send_fall_report`)에 열화상 이미지를 얹어 반복 전송해 실시간
스트리밍한다. 관찰 중이 아닐 때(대기 상태)는 보낼 위치 컨텍스트가 없으므로
아무것도 보내지 않는다.

`show=True`(main.py의 --show-thermal)면 관찰 중인 동안 로컬 디스플레이에도
컬러맵 창을 띄운다 — report_url 웹 스트리밍과 별개로, DISPLAY가 붙어있는
환경에서 바로 눈으로 확인하고 싶을 때 쓴다.
"""

import threading
import time

import cv2

from arda.utils import get_logger, send_fall_report

from . import thermal_backend as tb
from .bus import Bus, ThermalPan, ThermalVerdict

logger = get_logger(__name__)

JPEG_QUALITY = 85
TRIGGER_POLL_S = 0.2  # 트리거 대기 중 stop_event 확인 주기
PREEMPT_POLL_S = 0.02  # 관찰 중 "더 높은 확률의 새 트리거" 확인 주기
WINDOW_NAME = "ARDA Thermal — 관찰 중"


def run(
    bus: Bus,
    stop_event: threading.Event,
    backend: tb.Backend,
    read_frame_fn,
    i2c,
    dwell_seconds: float,
    required_consecutive: int,
    settle_offset: float,
    report_url: str,
    show: bool = False,
) -> None:
    """센서 초기화(`thermal_backend.initialize_sensor`)는 main.py가 스레드를
    띄우기 *전에* 미리 해둔다 — 실패 시(하드웨어 없음) 스레드 자체를 안
    띄우려면 main.py가 먼저 결과를 알아야 하고, I2C를 두 번 초기화하지
    않기 위해서이기도 하다(레이더의 /dev/ttyUSB* 존재 여부 사전 체크와
    같은 패턴)."""
    try:
        logger.info("열화상 센서 안정화 %d프레임 대기 중...", tb.SENSOR_WARMUP_FRAMES)
        for _ in range(tb.SENSOR_WARMUP_FRAMES):
            if stop_event.is_set():
                return
            tb.read_frame(read_frame_fn)
            time.sleep(tb.FRAME_INTERVAL_S)

        logger.info("열화상 대기 중 — 레이더 트리거를 기다립니다")
        trigger_ts = _wait_for_trigger(bus, stop_event)
        while trigger_ts is not None:
            latency_ms = (time.time() - trigger_ts) * 1000
            logger.info("열화상 트리거 수신(전송 후 %.0fms) — 최대 %.1fs 관찰 시작", latency_ms, dwell_seconds)

            person, next_trigger_ts = _run_observation(
                bus, stop_event, backend, read_frame_fn,
                dwell_seconds, required_consecutive, settle_offset, report_url, show,
            )

            if stop_event.is_set():
                break

            if person is None:
                # 더 높은 확률의 새 트리거로 대체됨 — 이미 받아둔 트리거로 바로 재관찰
                trigger_ts = next_trigger_ts
                continue

            logger.info("=" * 60)
            logger.info("열화상 판정 완료 — person=%s", person)
            logger.info("=" * 60)
            bus.verdict_q.put(ThermalVerdict(person=person, ts=time.time()))
            trigger_ts = _wait_for_trigger(bus, stop_event)
    finally:
        if show:
            cv2.destroyAllWindows()
        if i2c is not None and hasattr(i2c, "deinit"):
            i2c.deinit()

    logger.info("열화상 종료")


def _wait_for_trigger(bus: Bus, stop_event: threading.Event) -> float | None:
    """새 트리거 1건을 받을 때까지 짧은 타임아웃으로 반복 폴링. stop_event가
    켜지면 None을 반환해 상위 루프가 빠져나가게 한다."""
    while not stop_event.is_set():
        ts = bus.trigger_q.get(timeout=TRIGGER_POLL_S)
        if ts is not None:
            return ts
    return None


def _run_observation(
    bus: Bus,
    stop_event: threading.Event,
    backend: tb.Backend,
    read_frame_fn,
    dwell_seconds: float,
    required_consecutive: int,
    settle_offset: float,
    report_url: str,
    show: bool = False,
) -> tuple[bool | None, float | None]:
    """사람 모양 발열 영역이 required_consecutive 프레임 연속으로 잡히면
    (True, None)을, dwell_seconds 동안 못 잡으면(포기) (False, None)을 반환.

    더 높은 확률의 새 트리거가 오면 원칙적으로 (None, 새 트리거 ts)를 반환해
    호출부가 판정 없이 그 트리거로 즉시 재관찰을 시작하게 한다 — 단, 이번
    관찰에서 이미 발열 영역을 한 번이라도 검출했다면(bus.thermal_engaged)
    새 트리거를 무시하고 지금 추적을 계속한다. 레이더 좌표는 대략적인 초기
    조준일 뿐이고, 열화상이 실제 열원을 붙잡은 순간부터는 열화상이 우선권을
    갖는다는 원칙(arda_servo.ServoController._thermal_engaged와 동일)을
    여기서도 지켜야, 서보가 실제로 보고 있는 지점과 열화상이 판정하는
    지점이 어긋나지 않는다."""
    consecutive = 0
    frame_number = 0
    give_up_deadline: float | None = None
    bus.thermal_engaged.clear()

    while not stop_event.is_set() and (give_up_deadline is None or time.monotonic() < give_up_deadline):
        new_trigger_ts = bus.trigger_q.get(timeout=PREEMPT_POLL_S)
        if new_trigger_ts is not None:
            if bus.thermal_engaged.is_set():
                logger.info("[무시됨] 더 높은 확률의 낙하 후보 트리거 수신 — 이미 열원을 추적 중이라 무시하고 계속 관찰")
            else:
                logger.info("[대체됨] 더 높은 확률의 낙하 후보 트리거 수신 — 현재 관찰을 중단하고 즉시 재시작")
                return None, new_trigger_ts

        thermal = tb.read_frame(read_frame_fn)
        if thermal is None:
            continue
        frame_number += 1

        color_image = tb.create_absolute_colormap(thermal)
        detection = backend.detect(thermal, color_image)
        if detection.grid_xy is not None:
            bus.thermal_engaged.set()

        offset = None
        vertical_offset = None
        moving = False
        if detection.grid_xy is not None:
            gx, gy = detection.grid_xy
            offset = tb.offset_from_circle_x(gx)
            vertical_offset = tb.offset_from_circle_y(gy)
            bus.thermal_pan_q.put(ThermalPan(offset=offset, ts=time.time(), vertical_offset=vertical_offset))
            moving = abs(offset) > settle_offset

        if moving:
            give_up_deadline = None
        elif give_up_deadline is None:
            give_up_deadline = time.monotonic() + dwell_seconds

        if detection.matched:
            consecutive += 1
        elif not moving:
            consecutive = 0

        confirmed = detection.matched and consecutive >= required_consecutive

        logger.debug(
            "[관찰 %d] matched=%s consecutive=%d/%d moving=%s confirmed=%s",
            frame_number, detection.matched, consecutive, required_consecutive, moving, confirmed,
        )

        # 실시간 스트리밍 — 관찰(dwell) 중인 동안 매 프레임 기존 lat/lon/time
        # 리포트 포맷에 이미지를 얹어 반복 전송한다(사용자 지시). 관찰 중이
        # 아니면(pending_location 없음) 보낼 위치 컨텍스트가 없어 전송 안 함.
        loc = bus.pending_location.get()
        streaming = loc is not None and report_url
        if streaming or show:
            backend.draw(color_image, detection, confirmed)
        if streaming:
            ok, buf = cv2.imencode(".jpg", color_image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                send_fall_report(report_url, loc.lat, loc.lon, image_jpeg=buf.tobytes(), confirmed=confirmed)
        if show:
            cv2.imshow(WINDOW_NAME, color_image)
            cv2.waitKey(1)

        if confirmed:
            bus.thermal_pan_q.put(ThermalPan(
                offset=0.0, ts=time.time(), confirmed=True, vertical_offset=vertical_offset,
            ))
            logger.info("[열화상] 사람 확정 (frame=%d)", frame_number)
            return True, None

        time.sleep(tb.FRAME_INTERVAL_S)

    if stop_event.is_set():
        return None, None

    bus.thermal_pan_q.put(ThermalPan(offset=0.0, ts=time.time(), give_up=True))
    logger.info("[열화상] 관찰 실패 — 추적 포기 신호 전송")
    return False, None
