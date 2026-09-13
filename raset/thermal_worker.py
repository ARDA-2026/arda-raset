"""열화상(thermal-camera) 서브시스템 구동 — threshold/YOLO 공용.

thermal_main.py / thermal_main_yolo.py의 run_observation() 상태기계(대기
트리거 폴링 → 프레임 읽기 → 판정 → 서보 추적보정 전송 → settle 여부로
give-up 카운트다운 시작/보류 → 누적매칭 카운트 → 확정/포기 → 회신)를 그대로
이식하되, UDP 대신 bus 큐를 쓴다.

로컬 표시(show)와 웹 스트리밍(report_url)은 레이더 트리거를 기다리는
대기 상태에서도 상시로 이뤄진다 — 매 프레임(~0.5초 간격) 계속 읽어
표시/전송한다. 단, **판정(detect/draw)은 트리거가 와서 관찰(dwell) 중일
때만** 돌린다 — 대기 중에는 원본 컬러맵 이미지 그대로만 표시/전송한다
(오버레이 없음, 항상 confirmed=False). 아무도 안 지켜보는데 YOLO 등
무거운 판정을 상시로 돌리는 낭비를 막기 위함이다.

웹으로 보낼 때 실어야 하는 위치도 상태에 따라 다르다: 관찰 중에는 그
낙하의 실제 lat/lon(`bus.pending_location`)을, 대기 중에는 보낼 낙하
위치가 없으므로 설치 지점 좌표(site_lat/site_lon, `arda-radar`의
config/settings.yaml `site.lat/lon`)를 대신 싣는다. 이미지와 좌표를 별도
요청으로 쪼개지 않고, 기존 lat/lon/timestamp 리포트 포맷
(`arda.utils.send_fall_report`)에 이미지+confirmed를 얹어 한 번에
보낸다 — 요청을 둘로 쪼개는 것보다 가볍고 프로토콜도 그대로다.

`show=True`(main.py의 --show-thermal)면 대기/관찰 상태와 무관하게 항상
로컬 디스플레이에 컬러맵 창을 띄운다 — report_url 웹 스트리밍과 별개로,
DISPLAY가 붙어있는 환경에서 바로 눈으로 확인하고 싶을 때 쓴다.

웹/로컬로는 실제 온도 그대로인 절대 컬러맵(`create_absolute_colormap`)과
YOLO 입력용 배경-상대 보정 컬러맵(`create_relative_colormap`) 둘 다 매
프레임 같이 실어 보낸다(`thermal_image_base64`/`thermal_image_yolo_base64`,
`arda.utils.send_fall_report`) — 어느 쪽을 볼지는 재요청 없이 웹 쪽에서
그 자리에서 토글로 고른다. 판정(detect) 자체는 항상 각 백엔드가 내부적으로
쓰는 이미지로 고정돼 이 선택과 무관하게 전혀 바뀌지 않는다(ThresholdBackend는
raw thermal 배열만 보고, YoloBackend.detect()는 넘겨받은 color_image 인자를
무시하고 자기 안에서 create_relative_colormap을 다시 계산한다 —
thermal_backend.py 참고).
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
WINDOW_NAME = "ARDA Thermal"


def run(
    bus: Bus,
    stop_event: threading.Event,
    backend: tb.Backend,
    read_frame_fn,
    i2c,
    dwell_seconds: float,
    required_matches: int,
    settle_offset: float,
    report_url: str,
    show: bool = False,
    site_lat: float | None = None,
    site_lon: float | None = None,
) -> None:
    """센서 초기화(`thermal_backend.initialize_sensor`)는 main.py가 스레드를
    띄우기 *전에* 미리 해둔다 — 실패 시(하드웨어 없음) 스레드 자체를 안
    띄우려면 main.py가 먼저 결과를 알아야 하고, I2C를 두 번 초기화하지
    않기 위해서이기도 하다(레이더의 /dev/ttyUSB* 존재 여부 사전 체크와
    같은 패턴).

    site_lat/site_lon은 대기 상태(관찰 중이 아닐 때) 웹 스트리밍에 실을
    위치 — 낙하 위치가 아직 없으므로 설치 지점 좌표를 대신 쓴다."""
    try:
        logger.info("열화상 센서 안정화 %d프레임 대기 중...", tb.SENSOR_WARMUP_FRAMES)
        for _ in range(tb.SENSOR_WARMUP_FRAMES):
            if stop_event.is_set():
                return
            tb.read_frame(read_frame_fn)
            time.sleep(tb.FRAME_INTERVAL_S)

        logger.info("열화상 대기 중 — 레이더 트리거를 기다리며 상시 표시/스트리밍 중")
        trigger_ts = _wait_for_trigger(bus, stop_event, read_frame_fn, report_url, show, site_lat, site_lon)
        while trigger_ts is not None:
            latency_ms = (time.time() - trigger_ts) * 1000
            logger.info("열화상 트리거 수신(전송 후 %.0fms) — 최대 %.1fs 관찰 시작", latency_ms, dwell_seconds)

            person, next_trigger_ts = _run_observation(
                bus, stop_event, backend, read_frame_fn,
                dwell_seconds, required_matches, settle_offset, report_url, show,
                site_lat, site_lon,
            )

            if stop_event.is_set():
                break

            if person is None:
                # 더 높은 확률의 새 트리거로 대체됨 — 이미 받아둔 트리거로 바로 재관찰
                trigger_ts = next_trigger_ts
                continue

            logger.info("열화상 판정 완료 — person=%s", person)
            bus.verdict_q.put(ThermalVerdict(person=person, ts=time.time()))
            trigger_ts = _wait_for_trigger(bus, stop_event, read_frame_fn, report_url, show, site_lat, site_lon)
    finally:
        if show:
            cv2.destroyAllWindows()
        if i2c is not None and hasattr(i2c, "deinit"):
            i2c.deinit()

    logger.info("열화상 종료")


def _encode_jpeg(image) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None


def _publish_frame(
    backend: tb.Backend,
    absolute_image,
    yolo_image,
    detection: tb.Detection,
    confirmed: bool,
    report_url: str,
    lat: float | None,
    lon: float | None,
    show: bool,
) -> None:
    """판정 결과를 오버레이해 로컬 창(show)과 웹 스트리밍(report_url)에 반영.
    관찰(observation) 중에만 쓴다 — 대기 중(판정 없음)은 `_publish_idle_frame`.

    absolute_image(실제 온도)/yolo_image(YOLO 입력용 배경-상대 보정) 둘 다
    오버레이를 그려서 한 요청에 같이 실어 보낸다 — 웹 쪽이 재요청 없이
    그 자리에서 토글로 어느 쪽을 볼지 고른다. 로컬 창(show)은 absolute_image만
    띄운다(디버깅용 단일 창이라 굳이 둘 다 띄울 필요 없음).

    lat/lon이 없으면(웹 URL이 있어도) 전송하지 않는다."""
    streaming = bool(report_url) and lat is not None and lon is not None
    if not streaming and not show:
        return

    backend.draw(absolute_image, detection, confirmed)
    backend.draw(yolo_image, detection, confirmed)
    if streaming:
        abs_bytes = _encode_jpeg(absolute_image)
        yolo_bytes = _encode_jpeg(yolo_image)
        if abs_bytes is not None:
            send_fall_report(
                report_url, lat, lon, image_jpeg=abs_bytes, image_jpeg_yolo=yolo_bytes,
                confirmed=confirmed,
            )
    if show:
        cv2.imshow(WINDOW_NAME, absolute_image)
        cv2.waitKey(1)


def _publish_idle_frame(
    absolute_image,
    yolo_image,
    report_url: str,
    lat: float | None,
    lon: float | None,
    show: bool,
) -> None:
    """대기(트리거 없음) 중 — 판정(detect/draw)을 돌리지 않고 원본 컬러맵
    이미지 그대로만 로컬 창/웹에 반영한다(confirmed는 항상 False, 이미지+
    좌표를 한 요청에 얹어 보내는 기존 리포트 포맷 그대로). absolute_image/
    yolo_image 둘 다 같이 보낸다 — `_publish_frame` 참고.

    lat/lon이 없으면(웹 URL이 있어도) 전송하지 않는다."""
    streaming = bool(report_url) and lat is not None and lon is not None
    if not streaming and not show:
        return

    if streaming:
        abs_bytes = _encode_jpeg(absolute_image)
        yolo_bytes = _encode_jpeg(yolo_image)
        if abs_bytes is not None:
            send_fall_report(
                report_url, lat, lon, image_jpeg=abs_bytes, image_jpeg_yolo=yolo_bytes,
                confirmed=False,
            )
    if show:
        cv2.imshow(WINDOW_NAME, absolute_image)
        cv2.waitKey(1)


def _wait_for_trigger(
    bus: Bus,
    stop_event: threading.Event,
    read_frame_fn,
    report_url: str,
    show: bool,
    site_lat: float | None,
    site_lon: float | None,
) -> float | None:
    """새 트리거 1건을 받을 때까지 짧은 타임아웃으로 반복 폴링하되, show나
    report_url이 켜져 있으면 대기 중에도 매 프레임 읽어(판정 없이) 원본
    이미지를 로컬 표시/웹 스트리밍한다(위치는 설치 지점 좌표로 대체). 둘 다
    꺼져 있으면(기존 동작 그대로) 센서를 건드리지 않고 트리거만 기다린다.
    stop_event가 켜지면 None을 반환해 상위 루프가 빠져나가게 한다."""
    idle_publish = bool(show or report_url)
    while not stop_event.is_set():
        ts = bus.trigger_q.get(timeout=TRIGGER_POLL_S)
        if ts is not None:
            return ts
        if not idle_publish:
            continue

        thermal = tb.read_frame(read_frame_fn)
        if thermal is None:
            continue
        absolute_image = tb.create_absolute_colormap(thermal)
        yolo_image = tb.create_relative_colormap(thermal)
        _publish_idle_frame(absolute_image, yolo_image, report_url, site_lat, site_lon, show)
    return None


def _run_observation(
    bus: Bus,
    stop_event: threading.Event,
    backend: tb.Backend,
    read_frame_fn,
    dwell_seconds: float,
    required_matches: int,
    settle_offset: float,
    report_url: str,
    show: bool = False,
    site_lat: float | None = None,
    site_lon: float | None = None,
) -> tuple[bool | None, float | None]:
    """사람 모양 발열 영역이 이번 관찰(dwell) 동안 누적으로 required_matches
    번 잡히면(연속일 필요 없음) (True, None)을, dwell_seconds 동안 그만큼
    못 채우면(포기) (False, None)을 반환.

    더 높은 확률의 새 트리거가 오면 원칙적으로 (None, 새 트리거 ts)를 반환해
    호출부가 판정 없이 그 트리거로 즉시 재관찰을 시작하게 한다 — 단, 이번
    관찰에서 원하는 모양과 이미 한 번이라도 매칭됐다면(bus.thermal_engaged,
    즉 detection.matched=True가 처음 나온 시점부터 — required_matches
    중 1/N째) 새 트리거를 무시하고 지금 추적을 계속한다. 단순히 배경보다
    뜨거운 영역이 있다는 것만으로는(detection.grid_xy is not None이지만
    matched=False) engaged로 치지 않는다 — 사람 모양이 아닌 열원(반사광,
    손 등)에 선점권을 뺏기지 않기 위함이다. 레이더 좌표는 대략적인 초기
    조준일 뿐이고, 열화상이 원하는 모양과 매칭되기 시작한 순간부터는
    열화상이 우선권을 갖는다는 원칙(arda_servo.ServoController._thermal_engaged와
    동일)을 여기서도 지켜야, 서보가 실제로 보고 있는 지점과 열화상이 판정하는
    지점이 어긋나지 않는다."""
    match_count = 0
    frame_number = 0
    give_up_deadline: float | None = None
    bus.thermal_engaged.clear()

    while not stop_event.is_set() and (give_up_deadline is None or time.monotonic() < give_up_deadline):
        new_trigger_ts = bus.trigger_q.get(timeout=PREEMPT_POLL_S)
        if new_trigger_ts is not None:
            if bus.thermal_engaged.is_set():
                logger.info("[제어권 유지] 더 높은 확률의 낙하 후보 트리거 수신 — 이미 매칭된 대상을 추적 중이라 무시하고 계속 관찰")
            else:
                logger.info("[제어권 이동] 더 높은 확률의 낙하 후보 트리거 수신 — 현재 관찰을 중단하고 즉시 재시작")
                return None, new_trigger_ts

        thermal = tb.read_frame(read_frame_fn)
        if thermal is None:
            continue
        frame_number += 1

        absolute_image = tb.create_absolute_colormap(thermal)
        yolo_image = tb.create_relative_colormap(thermal)
        detection = backend.detect(thermal, absolute_image)
        if detection.matched:
            # 원하는 모양과 "처음" 매칭된 순간(=match_count가 1이 되는 시점)부터
            # 제어권을 넘긴다 — 그냥 열이 감지된 것만으로는(grid_xy) 넘기지 않는다.
            bus.thermal_engaged.set()

        offset = None
        vertical_offset = None
        moving = False
        if detection.grid_xy is not None:
            gx, gy = detection.grid_xy
            offset = tb.offset_from_circle_x(gx)
            vertical_offset = tb.offset_from_circle_y(gy)
            # matched를 함께 실어 보낸다 — arda_servo.ServoController도 이
            # 프레임이 원하는 모양과 매칭됐을 때만(_thermal_engaged) 서보
            # 제어권을 넘겨받도록 raset/thermal_worker.py와 동일한 기준을 쓴다.
            bus.thermal_pan_q.put(ThermalPan(
                offset=offset, ts=time.time(), vertical_offset=vertical_offset,
                matched=detection.matched,
            ))
            moving = abs(offset) > settle_offset

        if moving:
            give_up_deadline = None
        elif give_up_deadline is None:
            give_up_deadline = time.monotonic() + dwell_seconds

        if detection.matched:
            match_count += 1

        confirmed = detection.matched and match_count >= required_matches

        # "사람인지" 판정에 실제로 쓰이는 수치 중 YOLO의 confidence만 로그에
        # 추가로 보인다(threshold 백엔드는 대응하는 단일 수치가 없어
        # detection_detail()이 confidence를 안 채움 — 그대로 생략됨).
        conf = tb.detection_detail(detection).get("confidence")
        conf_note = f" conf={conf:.2f}" if conf is not None else ""

        logger.info(
            "[열화상 매칭시도 %d] matched=%s%s (%d/%d) confirmed=%s",
            frame_number, detection.matched, conf_note, match_count, required_matches, confirmed,
        )

        # 실시간 스트리밍 — 관찰(dwell) 중인 동안 매 프레임 기존 lat/lon/time
        # 리포트 포맷에 이미지를 얹어 반복 전송한다. 보통 pending_location에
        # 이번 낙하의 실제 위치가 있지만(radar_worker가 트리거 직전에 set),
        # 혹시 없는 경우에도(레이스 등) 상시 스트리밍이 끊기지 않도록 설치
        # 지점 좌표로 대체한다.
        loc = bus.pending_location.get()
        lat, lon = (loc.lat, loc.lon) if loc is not None else (site_lat, site_lon)
        _publish_frame(backend, absolute_image, yolo_image, detection, confirmed, report_url, lat, lon, show)

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
