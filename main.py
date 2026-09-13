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
  uv run python main.py --report-url http://172.30.139.125:8000/report
"""

import argparse
import logging
import os
import sys
import threading
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent

# 노트북(arda-algo_general, hanriver.py)의 웹 리포트 수신 주소 — 열화상
# 스트리밍(대기/관찰 중 이미지)과 낙하 확정 최종 좌표(서보 보정 후) 둘 다
# 이 값 하나로 통일해서 쓴다. 예전엔 arda-radar/config/settings.yaml과
# arda-servo/config/settings.yaml에 site.report_url이 각각 따로 있어서
# (전자는 스트리밍용, 후자는 확정 좌표용) 둘을 매번 손으로 맞춰야 했고
# 실제로 어긋난 적도 있었다 — 이제 raset은 그 두 값을 더 이상 읽지 않고
# 여기 하나만 본다(그 두 저장소를 raset 없이 단독 실행할 때는 각자 자기
# 파일의 값을 계속 그대로 씀).
#
# 핫스팟은 재연결할 때마다 IP가 바뀔 수 있다 — 바뀌면 아래 값을 고치거나,
# 재빌드/파일 수정 없이 그때그때 --report-url로 덮어써도 된다.
DEFAULT_REPORT_URL = "http://10.70.108.191:8000/report"

# 설치 지점 위도/경도/방위각 — report_url과 같은 이유로 하나로 통일한다.
# 예전엔 arda-radar/config/settings.yaml과 arda-servo/config/settings.yaml에
# site.lat/lon/heading_deg가 각각 따로 있어서(전자는 레이더 좌표계 변환/
# 열화상 대기 중 스트리밍 위치용, 후자는 서보의 낙하 확정 최종 위경도
# 계산용) 둘을 매번 손으로 맞춰야 했다 — 이제 raset은 그 두 값을 더 이상
# 읽지 않고 여기 하나만 본다(그 두 저장소를 raset 없이 단독 실행할 때는
# 각자 자기 파일의 값을 계속 그대로 씀).
DEFAULT_SITE_LAT = 37.5336        # 설치 지점 위도 (도, WGS84)
DEFAULT_SITE_LON = 126.9364        # 설치 지점 경도 (도, WGS84)
DEFAULT_SITE_HEADING_DEG = 320.0   # 센서 정면(Y축)이 향하는 나침반 방위각 (정북 0°, 시계방향 동 90°)


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
    parser.add_argument(
        "--dwell-margin-seconds", type=float, default=30.0,
        help="thermal_pending_timeout과 servo_dwell_seconds가 자동(-1)일 때 공통으로 쓰는 "
             "안전 마진(초) — dwell_seconds + 이 값을 상한으로 삼는다. 실기 테스트로 40초 "
             "이상 걸리는 경우도 있어 기본을 30.0으로 잡음(arda-bringup의 dwell_margin_seconds와 동일)",
    )
    parser.add_argument(
        "--thermal-pending-timeout", type=float, default=-1.0,
        help="레이더가 열화상 판정을 기다리는 최대 시간(초). -1(기본)=자동으로 "
             "dwell_seconds+dwell_margin_seconds 사용 — 이 값이 dwell_seconds와 같거나 "
             "작으면 열화상이 실제 관찰을 마치기 전에 레이더가 먼저 포기해 verdict가 "
             "유실되는 실측 버그가 있었다. 양수를 명시하면 그 값을 그대로 쓰되, "
             "dwell_seconds보다 충분히 크게 줄 것.",
    )
    parser.add_argument(
        "--servo-dwell-seconds", type=float, default=-1.0,
        help="서보가 dwell 중 마지막 조준 각도에서 버티는 시간(초) — 열화상 관찰 시간"
             "(dwell_seconds)과는 별개 값. -1(기본)=자동으로 "
             "max(arda-servo/config/settings.yaml의 servo.dwell_seconds, "
             "dwell_seconds+dwell_margin_seconds) 사용 — yaml 기본값을 무조건 신뢰하면 "
             "dwell_seconds보다 작아서 서보가 열화상보다 먼저 포기해버리는 실측 버그가 "
             "있었다. 양수를 명시하면 그 값을 그대로 쓰되, dwell_seconds보다 작으면 경고만 남는다.",
    )
    parser.add_argument(
        "--report-url", default=None,
        help=f"노트북(hanriver.py) 웹 리포트 수신 주소 — 열화상 스트리밍과 낙하 확정 "
             f"최종 좌표(서보 보정 후) 둘 다 이 값으로 통일해서 보낸다. 생략하면 "
             f"DEFAULT_REPORT_URL(현재: {DEFAULT_REPORT_URL}) 사용. 빈 문자열('')을 "
             f"주면 전송 자체를 끈다.",
    )
    parser.add_argument(
        "--site-lat", type=float, default=None,
        help=f"설치 지점 위도(도) — 레이더 좌표계 변환/열화상 스트리밍 위치/서보 낙하 "
             f"확정 위경도 계산에 전부 이 값 하나로 통일해서 쓴다. 생략하면 "
             f"DEFAULT_SITE_LAT(현재: {DEFAULT_SITE_LAT}) 사용.",
    )
    parser.add_argument(
        "--site-lon", type=float, default=None,
        help=f"설치 지점 경도(도). 생략하면 DEFAULT_SITE_LON(현재: {DEFAULT_SITE_LON}) 사용.",
    )
    parser.add_argument(
        "--site-heading-deg", type=float, default=None,
        help=f"센서 정면(Y축)이 향하는 나침반 방위각(도, 정북 0°, 시계방향 동 90°). "
             f"생략하면 DEFAULT_SITE_HEADING_DEG(현재: {DEFAULT_SITE_HEADING_DEG}) 사용.",
    )
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
    from arda.utils import get_logger
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

    # report_url/site_lat/site_lon/site_heading_deg는 arda-radar나 arda-servo의
    # config/settings.yaml이 아니라 이 파일 상단의 DEFAULT_*/--report-url·
    # --site-lat·--site-lon·--site-heading-deg에서 가져온다 — 저장소마다
    # 따로 있던 값을 하나로 통일한 것(위 DEFAULT_REPORT_URL/DEFAULT_SITE_LAT
    # 주석 참고).
    report_url = args.report_url if args.report_url is not None else DEFAULT_REPORT_URL
    site_lat = args.site_lat if args.site_lat is not None else DEFAULT_SITE_LAT
    site_lon = args.site_lon if args.site_lon is not None else DEFAULT_SITE_LON
    site_heading_deg = args.site_heading_deg if args.site_heading_deg is not None else DEFAULT_SITE_HEADING_DEG
    if report_url:
        logger.info("웹 리포트 전송 활성화 — %s (열화상 상시 스트리밍 포함, 대기 중엔 설치 지점 좌표 사용)", report_url)
    else:
        logger.info("report_url이 비어 있어 웹 리포트/열화상 스트리밍은 전송되지 않습니다")

    with open(servo_config_path, encoding="utf-8") as f:
        servo_cfg = yaml.safe_load(f)

    # servo_dwell_seconds 처리 — 명시적으로 준 값(>=0)은 그대로 쓰되
    # dwell_seconds보다 작으면 경고만 남긴다(사용자 의도 존중). 자동(-1,
    # 기본)이면 yaml 값을 무조건 신뢰하지 않고 dwell_seconds+dwell_margin_seconds와
    # 비교해 더 큰 쪽을 쓴다 — yaml 기본값(10.0)이 dwell_seconds보다 작으면
    # 열화상이 아직 관찰 중인데 서보 자신의 폴백 타이머가 먼저 만료돼버려
    # "매칭시도 로그는 계속 찍히는데 서보는 dwell 초과로 끝남" 현상이
    # 실측됨(서보 dwell 연장은 grid_xy가 있는 프레임에서만 일어나고, 매칭시도
    # 로그는 grid_xy 유무와 무관하게 매 프레임 찍히므로 로그가 계속 나온다고
    # 서보가 보정을 받고 있다는 뜻은 아니다). arda-bringup의 tracker_node.py
    # _start_raset()와 동일한 로직.
    yaml_servo_dwell_seconds = servo_cfg.get("servo", {}).get("dwell_seconds", 10.0)
    if args.servo_dwell_seconds >= 0.0:
        effective_servo_dwell_seconds = args.servo_dwell_seconds
        if effective_servo_dwell_seconds < args.dwell_seconds:
            logger.warning(
                "servo_dwell_seconds(%.1fs)가 dwell_seconds(%.1fs)보다 작습니다 — 열화상이 "
                "아직 관찰 중인데 서보가 먼저 포기하고 각도를 고정할 수 있습니다.",
                effective_servo_dwell_seconds, args.dwell_seconds,
            )
    else:
        effective_servo_dwell_seconds = max(
            yaml_servo_dwell_seconds, args.dwell_seconds + args.dwell_margin_seconds,
        )
    servo_cfg.setdefault("servo", {})["dwell_seconds"] = effective_servo_dwell_seconds

    # 추가로, 열화상 프레임 주기(FRAME_INTERVAL_S)보다도 충분히(2배 이상)
    # 커야 한다 — grid_xy가 매 프레임 검출되는 정상 상황에서도 "추적 연장"이
    # 다음 보정 도착 전에 만료되지 않게 하기 위함. 위 자동 계산이면 사실상
    # 항상 만족되지만, 명시적으로 아주 작은 값을 준 경우를 대비해 검사한다.
    frame_interval_s = thermal_backend.FRAME_INTERVAL_S
    if effective_servo_dwell_seconds <= frame_interval_s:
        logger.error(
            "servo_dwell_seconds(%.1fs)가 열화상 프레임 주기(%.1fs) 이하입니다 — 서보가 "
            "열화상 관찰 도중에 dwell이 끝나 추적을 멈출 수 있습니다. servo_dwell_seconds를 늘리세요.",
            effective_servo_dwell_seconds, frame_interval_s,
        )
    elif effective_servo_dwell_seconds < frame_interval_s * 2:
        logger.warning(
            "servo_dwell_seconds(%.1fs)가 열화상 프레임 주기(%.1fs)의 2배보다 작습니다 — "
            "프레임 지연이 조금만 있어도 추적 연장이 늦어 dwell이 만료될 수 있습니다.",
            effective_servo_dwell_seconds, frame_interval_s,
        )

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
    # report_url/site_lat/site_lon/site_heading_deg는 arda-servo/config/
    # settings.yaml이 아니라 위에서 정한 통일된 값을 그대로 넘긴다 —
    # 낙하 확정 최종 좌표도 열화상 스트리밍과 같은 주소/기준점으로 나가도록.
    _spawn(
        "servo", servo_worker.run, bus, stop_event, servo_cfg, args.simulate_servo,
        report_url, site_lat, site_lon, site_heading_deg,
    )

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

    # thermal_pending_timeout: radar_worker가 열화상 verdict를 기다리는 최대
    # 시간. 열화상의 실제 관찰 소요시간은 dwell_seconds 그 자체가 아니라
    # 항상 그보다 더 걸린다(트리거 전파 지연, give-up 판정이 다음 프레임
    # 루프에서만 감지되는 지연, 열원이 계속 settle_offset 밖에서 움직이는
    # 동안 dwell 카운트다운 자체가 계속 리셋되는 unbounded 케이스 등). 이
    # 값이 dwell_seconds와 같거나 비슷하면 radar_worker가 먼저 timeout
    # 처리를 해버려 pending_location을 지우고, 뒤늦게 도착한 열화상
    # verdict는 조용히 버려진다 — absolute_pose/report_url 전송이 안 되는
    # 근본 원인이었다. arda-bringup의 tracker_node.py와 동일한 자동 계산.
    thermal_pending_timeout = (
        args.thermal_pending_timeout if args.thermal_pending_timeout > 0.0
        else args.dwell_seconds + args.dwell_margin_seconds
    )

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
            thermal_started, thermal_pending_timeout, site_lat, site_lon, site_heading_deg,
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
