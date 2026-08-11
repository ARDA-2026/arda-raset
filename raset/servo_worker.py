"""서보(arda_servo) 서브시스템 구동.

arda_servo.controller.ServoController.step()을 원본 그대로 재사용한다 —
duck-typed `receiver`/`thermal_receiver`만 UDP 대신 bus 큐를 감싼
QueueReceiver로 바뀌었을 뿐, 서보 각도 계산·dwell·선점·열화상 추적 로직은
1바이트도 손대지 않는다.

`ServoController.run_forever()`는 내부에서 `while True`를 돌며
KeyboardInterrupt를 자체적으로 잡는데, 백그라운드 스레드는 SIGINT를 못
받으므로 재사용할 수 없다 — 대신 `run_forever()`가 시작 시 하던 일(홈
포지션으로 1회 이동)만 그대로 재현하고, 직접 만든 stop_event 기반 루프에서
`step()`을 반복 호출한다.
"""

import threading

from arda_servo.controller import ServoController
from arda_servo.servo import PanServo
from arda_servo.utils import get_logger

from .bus import Bus, QueueReceiver

logger = get_logger(__name__)


def run(bus: Bus, stop_event: threading.Event, cfg: dict, simulate: bool) -> None:
    servo_cfg = cfg["servo"]

    servo = PanServo(
        pin=servo_cfg["gpio_pin"],
        min_deg=servo_cfg.get("min_deg", 0.0),
        max_deg=servo_cfg.get("max_deg", 180.0),
        simulate=simulate,
    )

    offset_cfg = cfg.get("mount_offset", {})
    offset_x = offset_cfg.get("x", 0.0)
    offset_y = offset_cfg.get("y", 0.0)
    offset_z = offset_cfg.get("z", 0.0)
    if offset_z:
        logger.info(
            "mount_offset.z=%.2fm 설정됨 — 팬(pan) 각도 계산에는 미반영",
            offset_z,
        )

    thermal_cfg = cfg.get("thermal_udp") or {}
    geometry_cfg = cfg.get("camera_geometry")
    install_height_m = camera_tilt_deg = vertical_fov_deg = None
    if geometry_cfg:
        install_height_m = (
            geometry_cfg.get("radar_height_m", 0.0)
            + geometry_cfg.get("height_offset_from_radar_m", 0.0)
        )
        camera_tilt_deg = geometry_cfg.get("tilt_deg")
        vertical_fov_deg = geometry_cfg.get("vertical_fov_deg")

    site_cfg = cfg.get("site") or {}

    # bus.coord_q는 레이더 스레드가 아예 안 뜨는 경우(USB 없음/--no-radar)에도
    # 항상 존재한다 — ServoController.step()이 self._receiver.recv()를
    # 무조건(None 체크 없이) 호출하므로 receiver=None이면 첫 스텝에서 죽는다.
    coord_receiver = QueueReceiver(bus.coord_q, timeout=0.5)
    thermal_receiver = QueueReceiver(bus.thermal_pan_q, timeout=0.01)

    center_deg = servo_cfg.get("center_deg", 90.0)
    controller = ServoController(
        servo,
        coord_receiver,
        center_deg=center_deg,
        invert=servo_cfg.get("invert", False),
        offset_x=offset_x,
        offset_y=offset_y,
        dwell_seconds=servo_cfg.get("dwell_seconds", 10.0),
        thermal_receiver=thermal_receiver,
        thermal_pan_gain_deg=thermal_cfg.get("pan_gain_deg", 8.0),
        install_height_m=install_height_m,
        camera_tilt_deg=camera_tilt_deg,
        vertical_fov_deg=vertical_fov_deg,
        site_lat=site_cfg.get("lat"),
        site_lon=site_cfg.get("lon"),
        site_heading_deg=site_cfg.get("heading_deg", 0.0),
    )

    servo.set_angle(center_deg)  # run_forever()가 시작 시 하던 일 그대로 재현
    logger.info("서보 제어기 시작 — 홈 포지션(%.1f°)에서 낙하 트리거 대기 중", center_deg)
    try:
        while not stop_event.is_set():
            controller.step()
    finally:
        servo.close()
