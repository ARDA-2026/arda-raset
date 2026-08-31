#!/usr/bin/env python3
"""arda-raset 진입점 — arda-radar / arda-servo / thermal-camera를 UDP가
아니라 하나의 Python 프로세스 안에서 스레드 3개 + 인메모리 큐로 통합 실행한다.

세 프로젝트는 sys.path 삽입으로 라이브러리처럼 import해서 재사용한다(코드
복제 없음, 각 저장소가 여전히 단일 소스). 기본적으로 이 파일과 같은 부모
디렉터리(ARDA-2026/) 아래 형제 디렉터리로 있다고 가정하며, 다른 경로에
있으면 ARDA_RADAR_DIR / ARDA_SERVO_DIR / ARDA_THERMAL_DIR 환경변수로
덮어쓸 수 있다.

사용법 예시:
  uv run python main.py                                # 전부 실제 하드웨어
  uv run python main.py --simulate-servo --simulate-thermal --no-radar
  uv run python main.py --yolo                          # 열화상 판정을 YOLO로
  uv run python main.py --show-thermal                  # 열화상 컬러맵 창을 로컬에 상시 표시
"""

import argparse
import logging
import os
import sys
import threading
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def _setup_logging(log_dir: Path) -> None:
    """arda/arda_servo의 로거 모듈이 각자 import 시점에 logging.basicConfig()를
    무조건 호출해 서로의 파일 핸들러를 밀어내는 경쟁이 있다 — 그 모듈들을
    import하기 전에 우리가 먼저 basicConfig()를 호출해두면(force 없이도)
    그쪽 호출은 전부 no-op이 되어 이 설정이 그대로 유지된다."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "arda-raset.log", encoding="utf-8"),
        ],
    )


def _setup_sys_path() -> tuple[Path, Path, Path]:
    radar_dir = Path(os.environ.get("ARDA_RADAR_DIR", ROOT.parent / "arda-radar")).resolve()
    servo_dir = Path(os.environ.get("ARDA_SERVO_DIR", ROOT.parent / "arda-servo")).resolve()
    thermal_dir = Path(os.environ.get("ARDA_THERMAL_DIR", ROOT.parent / "thermal-camera")).resolve()
    for d, label in ((radar_dir, "arda-radar"), (servo_dir, "arda-servo"), (thermal_dir, "thermal-camera")):
        if not d.is_dir():
            raise SystemExit(
                f"{label} 디렉터리를 찾을 수 없습니다: {d}\n"
                "ARDA_RADAR_DIR / ARDA_SERVO_DIR / ARDA_THERMAL_DIR 환경변수로 경로를 지정하세요."
            )
    sys.path.insert(0, str(radar_dir))  # -> import arda
    sys.path.insert(0, str(servo_dir))  # -> import arda_servo
    sys.path.insert(0, str(thermal_dir / "src"))  # -> import thermal_main / thermal_main_yolo
    return radar_dir, servo_dir, thermal_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="arda-raset — arda-radar/arda-servo/thermal-camera 단일 프로세스 통합 실행",
    )
    parser.add_argument("--simulate-servo", action="store_true", help="서보 GPIO 없이 각도 계산만")
    parser.add_argument("--simulate-thermal", action="store_true", help="열화상 센서 없이 임의 프레임")
    parser.add_argument("--no-radar", action="store_true", help="레이더 강제 생략")
    parser.add_argument("--yolo", action="store_true", help="열화상 판정을 YOLO 백엔드로 (기본: threshold)")
    parser.add_argument("--show-thermal", action="store_true", help="열화상 컬러맵 창을 로컬 디스플레이에 상시 표시 — 대기 중에도 계속 (DISPLAY 환경변수 필요)")
    parser.add_argument("--model-path", default=None, help="YOLO 모델(.pt) 경로 (기본: thermal-camera/models/s_yolo26.pt)")
    parser.add_argument("--confidence-threshold", type=float, default=0.4, help="YOLO 검출 신뢰도 임계값")
    parser.add_argument("--device", default="cuda", help="YOLO 추론 디바이스 ('cuda' 또는 'cpu')")
    parser.add_argument("--dwell-seconds", type=float, default=10.0, help="열화상 트리거 후 최대 관찰 시간(초)")
    parser.add_argument("--required-matches", type=int, default=3, help="열화상 확정에 필요한 누적 매칭 횟수(연속일 필요 없음)")
    parser.add_argument("--settle-offset", type=float, default=0.15, help="서보 settle 판단 기준 (정규화 -1.0~1.0)")
    parser.add_argument("--thermal-pending-timeout", type=float, default=10.0, help="레이더가 열화상 판정을 기다리는 최대 시간(초)")
    parser.add_argument("--radar-cli-port", default="/dev/ttyUSB0", help="레이더 CLI 시리얼 포트")
    parser.add_argument("--radar-data-port", default="/dev/ttyUSB1", help="레이더 데이터 시리얼 포트")
    parser.add_argument("--radar-settings", default=None, help="기본: <arda-radar>/config/settings.yaml")
    parser.add_argument("--radar-profile", default=None, help="기본: <arda-radar>/config/profiles/xwr68xx_AOP_profile_short_range.cfg")
    parser.add_argument("--servo-config", default=None, help="기본: <arda-servo>/config/settings.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    _setup_logging(ROOT / ".logs")
    radar_dir, servo_dir, thermal_dir = _setup_sys_path()

    # sys.path가 준비되고 우리 로깅 설정이 먼저 걸린 뒤에야 arda/arda_servo/
    # thermal_main을 import하는 raset 서브모듈들을 불러온다.
    from arda.utils import get_logger, load_settings
    from raset import radar_worker, servo_worker, thermal_backend, thermal_worker
    from raset.bus import Bus

    logger = get_logger("raset.main")

    radar_settings_path = (
        Path(args.radar_settings) if args.radar_settings else radar_dir / "config" / "settings.yaml"
    )
    radar_profile_path = (
        Path(args.radar_profile) if args.radar_profile
        else radar_dir / "config" / "profiles" / "xwr68xx_AOP_profile_short_range.cfg"
    )
    servo_config_path = (
        Path(args.servo_config) if args.servo_config else servo_dir / "config" / "settings.yaml"
    )

    site_cfg = load_settings(radar_settings_path).get("site", {})
    report_url = site_cfg.get("report_url", "")
    site_lat = site_cfg.get("lat")
    site_lon = site_cfg.get("lon")
    if report_url:
        logger.info("웹 리포트 전송 활성화 — %s (열화상 상시 스트리밍 포함, 대기 중엔 설치 지점 좌표 사용)", report_url)
    else:
        logger.info("site.report_url이 비어 있어 웹 리포트/열화상 스트리밍은 전송되지 않습니다")

    with open(servo_config_path, encoding="utf-8") as f:
        servo_cfg = yaml.safe_load(f)

    bus = Bus()
    stop_event = threading.Event()
    threads: list[threading.Thread] = []

    def _spawn(name: str, target, *fn_args) -> None:
        def _wrapped() -> None:
            try:
                target(*fn_args)
            except Exception:
                logger.exception("[%s] 처리되지 않은 예외로 종료 — 전체를 정리하고 내려갑니다", name)
                stop_event.set()

        t = threading.Thread(target=_wrapped, name=name, daemon=True)
        t.start()
        threads.append(t)

    # 서보는 항상 기동한다 — PanServo가 Jetson.GPIO를 못 찾으면 자동으로
    # 시뮬레이션 모드로 떨어지므로 하드웨어 부재를 걱정할 필요가 없다.
    _spawn("servo", servo_worker.run, bus, stop_event, servo_cfg, args.simulate_servo)

    # 열화상 — 센서 초기화를 스레드를 띄우기 전에 미리 해서, 실패 시(하드웨어
    # 없음, --simulate-thermal 아님) 스레드 자체를 안 띄운다. YOLO 모델 로드
    # 실패(잘못된 --model-path 등)도 마찬가지로 여기서 미리 잡는다.
    show_thermal = args.show_thermal
    if show_thermal and not os.environ.get("DISPLAY"):
        logger.info("DISPLAY 환경변수가 없어 --show-thermal을 무시합니다")
        show_thermal = False

    thermal_started = False
    try:
        i2c, read_frame_fn = thermal_backend.initialize_sensor(simulate=args.simulate_thermal)
    except RuntimeError as exc:
        logger.info("열화상 센서를 사용할 수 없어 생략합니다: %s", exc)
    else:
        try:
            if args.yolo:
                model_path = args.model_path or str(thermal_dir / "models" / "s_yolo26.pt")
                backend = thermal_backend.YoloBackend(model_path, args.device, args.confidence_threshold)
            else:
                backend = thermal_backend.ThresholdBackend()
        except Exception as exc:  # noqa: BLE001 — YOLO 모델/torch 로드 실패 등 다양한 예외
            logger.error("열화상 판정 백엔드 초기화 실패 — 열화상 없이 진행합니다: %s", exc)
            if i2c is not None and hasattr(i2c, "deinit"):
                i2c.deinit()
        else:
            _spawn(
                "thermal", thermal_worker.run, bus, stop_event, backend, read_frame_fn, i2c,
                args.dwell_seconds, args.required_matches, args.settle_offset, report_url,
                show_thermal, site_lat, site_lon,
            )
            thermal_started = True

    # 레이더 — USB 시리얼 포트가 없으면 생략(자동 감지, 기존 run_all.sh와 동일 UX).
    if args.no_radar:
        logger.info("--no-radar 옵션으로 레이더 생략")
    elif not (Path(args.radar_cli_port).exists() and Path(args.radar_data_port).exists()):
        logger.info(
            "%s, %s 가 없어 레이더는 생략합니다 (USB 연결 확인)",
            args.radar_cli_port, args.radar_data_port,
        )
    else:
        _spawn(
            "radar", radar_worker.run, bus, stop_event,
            args.radar_cli_port, args.radar_data_port, radar_profile_path, radar_settings_path,
            thermal_started, args.thermal_pending_timeout,
        )

    logger.info("arda-raset 전부 시작됨 (Ctrl+C로 종료)")
    try:
        stop_event.wait()
    except KeyboardInterrupt:
        logger.info("사용자 중단")
    finally:
        stop_event.set()
        join_timeout = 2.0
        for t in threads:
            t.join(timeout=join_timeout)
            if t.is_alive():
                logger.warning(
                    "[%s] 스레드가 %.1fs 안에 끝나지 않았습니다 — 데몬 스레드라 "
                    "프로세스 종료 시 함께 정리됩니다", t.name, join_timeout,
                )
        logger.info("arda-raset 종료")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
