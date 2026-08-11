"""레이더(arda-radar) 서브시스템 구동.

arda-radar/main.py의 main() 루프를 그대로 이식한다 — 센서 프레임 읽기 →
클러스터링 → FallDetector.update() → 낙하 확정 시 dedup·좌표변환·GPS 로그·
report_url POST·열화상 게이트의 pending/preemption/timeout 로직까지 거의
동일하게 유지한다. 바뀐 건 UDP CoordSender/ThermalTriggerSender/
ThermalVerdictReceiver가 bus 큐로 교체된 것, 열화상 트리거를 보내는/거두는
시점에 bus.pending_location도 같이 세팅/해제하는 것(실시간 스트리밍이 읽는
"지금 관찰 중인 낙하의 위치"), 그리고 confidence 기반 선점 조건에
bus.thermal_engaged 체크가 추가된 것이다 — 열화상이 이미 열원을 붙잡아
추적 중이면(arda_servo.ServoController._thermal_engaged와 같은 원칙)
confidence가 더 높은 새 후보가 와도 선점하지 않는다(arda-radar/main.py의
원본 로직은 이 체크가 없어 별도로 수정함, arda-radar 참고).

RealtimePlotter는 쓰지 않는다 — matplotlib GUI 이벤트 루프는 메인 스레드
전용이라 통합 프로세스(백그라운드 스레드)에서는 안전하지 않다. 3D 플롯이
필요하면 arda-radar를 단독 실행할 것.
"""

import threading
import time
from pathlib import Path

from arda.detection import FallDetector
from arda.processing import PointCloud, cluster_points
from arda.processing.filters import filter_stationary
from arda.radar import IWR6843Sensor
from arda.utils import (
    get_logger,
    load_processing_config,
    load_settings,
    local_to_latlon,
    send_fall_report,
)

from .bus import Bus, Coord

logger = get_logger(__name__)


def run(
    bus: Bus,
    stop_event: threading.Event,
    cli_port: str,
    data_port: str,
    profile_path: Path,
    settings_path: Path,
    thermal_gate: bool,
    thermal_pending_timeout: float,
    report_url: str,
) -> None:
    settings = load_settings(settings_path)
    site_cfg = settings.get("site", {})
    site_lat = site_cfg.get("lat", 0.0)
    site_lon = site_cfg.get("lon", 0.0)
    site_heading_deg = site_cfg.get("heading_deg", 0.0)

    cfg = load_processing_config(settings_path)

    sensor = IWR6843Sensor(cli_port, data_port)
    sensor.configure(profile_path)

    detector = FallDetector(max_jump=cfg["max_jump"])

    # 열화상 게이트 대기 중인 낙하 1건의 위도/경도·신뢰도와 트리거 전송 시각.
    # 응답이 오거나 pending-timeout이 지나면 None으로 비운다.
    pending_latlon = None
    pending_confidence = 0.0
    pending_since = 0.0
    # FallDetector.update()는 한 번 확정된 트랙에 계속 True를 반환하므로(래치),
    # 마지막으로 반응한 트랙 id를 기억해 "새로 확정된 낙하"일 때만 반응한다.
    last_reacted_track_id = None

    logger.info("레이더 시작 — 낙하 감지 모니터링 중 (thermal_gate=%s)", thermal_gate)
    with sensor:
        while not stop_event.is_set():
            frame = sensor.read_frame()
            if frame is None:
                continue

            pc = PointCloud(frame["points"])
            pc = (pc.filter_snr(min_snr=cfg["min_snr"])
                    .filter_roi(x_range=cfg["roi_x"], y_range=cfg["roi_y"], z_range=cfg["roi_z"]))
            pc = filter_stationary(pc, min_abs_doppler=cfg["min_abs_doppler"])

            clusters = cluster_points(pc, eps=cfg["cluster_eps"], min_samples=cfg["cluster_min_samples"])

            fell = detector.update(clusters)

            if (
                fell
                and detector.last_fall_centroid is not None
                and detector.last_fall_track_id != last_reacted_track_id
            ):
                last_reacted_track_id = detector.last_fall_track_id

                x, y, z = detector.last_fall_centroid
                bus.coord_q.put(Coord(
                    x=float(x), y=float(y), z=float(z),
                    fall=True, confidence=float(detector.last_fall_confidence), ts=time.time(),
                ))

                lat, lon = local_to_latlon(x, y, site_lat, site_lon, site_heading_deg)

                logger.warning("*" * 50)
                logger.warning("레이더 낙하 판단 — 로컬 X=%.2f Y=%.2f", x, y)
                logger.warning("*" * 50)

                if not thermal_gate:
                    logger.warning("낙하 위치(GPS) lat=%.6f lon=%.6f", lat, lon)
                elif pending_latlon is None:
                    # 대기 중인 낙하가 없으면 바로 트리거.
                    bus.trigger_q.put(time.time())
                    bus.pending_location.set(lat, lon)
                    pending_latlon = (lat, lon)
                    pending_confidence = detector.last_fall_confidence
                    pending_since = time.time()
                    logger.info(
                        "열화상 판정 요청 전송 — lat=%.6f lon=%.6f confidence=%.2f, 회신 대기 중",
                        lat, lon, pending_confidence,
                    )
                elif bus.thermal_engaged.is_set():
                    # 열화상이 이미 대기 중인 낙하의 열원을 붙잡아 추적 중이다 —
                    # confidence와 무관하게 선점하지 않는다. arda_servo의
                    # ServoController._thermal_engaged와 같은 원칙: 열화상이
                    # 실제 열원을 붙잡은 순간부터는 열화상이 우선권을 갖는다.
                    logger.debug(
                        "더 높은 확률의 낙하 후보(%.2f > %.2f) 발견했지만 열화상이 이미 열원을 "
                        "추적 중이라 무시함 — 열화상이 우선권을 가짐",
                        detector.last_fall_confidence, pending_confidence,
                    )
                elif detector.last_fall_confidence > pending_confidence:
                    # 이미 대기 중인 낙하보다 confidence가 더 높다 — 기존 대기를
                    # 포기하고 이 후보로 즉시 대체한다.
                    logger.info(
                        "더 높은 확률의 낙하 후보 발견(%.2f > %.2f) — 기존 판정 대기 취소, "
                        "새 트리거 전송 lat=%.6f lon=%.6f",
                        detector.last_fall_confidence, pending_confidence, lat, lon,
                    )
                    bus.trigger_q.put(time.time())
                    bus.pending_location.set(lat, lon)
                    pending_latlon = (lat, lon)
                    pending_confidence = detector.last_fall_confidence
                    pending_since = time.time()

            if thermal_gate:
                verdict = bus.verdict_q.get(timeout=0.01)
                if verdict is not None and pending_latlon is not None:
                    vlat, vlon = pending_latlon
                    if verdict.person:
                        logger.warning("낙하 위치(GPS) lat=%.6f lon=%.6f — 열화상 확인됨", vlat, vlon)
                        if report_url:
                            send_fall_report(report_url, vlat, vlon)
                    else:
                        logger.info("낙하 판정 기각 — 열화상에서 사람 미확인 (lat=%.6f lon=%.6f)", vlat, vlon)
                    pending_latlon = None
                    pending_confidence = 0.0
                    bus.pending_location.clear()
                elif (
                    pending_latlon is not None
                    and (time.time() - pending_since) > thermal_pending_timeout
                ):
                    logger.warning(
                        "열화상 판정 응답 없음(%.1fs 경과) — 낙하 확정 보류",
                        time.time() - pending_since,
                    )
                    pending_latlon = None
                    pending_confidence = 0.0
                    bus.pending_location.clear()

    logger.info("레이더 종료")
