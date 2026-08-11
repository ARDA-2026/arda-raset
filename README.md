# arda-raset

`arda-radar` + `arda-servo` + `thermal-camera`를 **하나의 Python 프로세스**
안에서 통합 실행하는 프로그램. 세 프로젝트는 원래 각자 별도 OS 프로세스로
떠서 localhost UDP로 통신했는데, arda-raset은 그 UDP 링크를 전부 인메모리
큐로 대체하고 세 서브시스템을 스레드로 띄워 한 프로세스 안에서 직접
데이터를 주고받는다(네트워크 왕복 없음).

세 저장소의 검증된 로직 — 레이더 신호 처리·낙하 판정, 서보 각도 계산·
dwell/선점, 열화상 발열 판정 — 은 재작성하지 않고 라이브러리처럼 그대로
import해서 재사용한다(`sys.path`로 각 저장소를 경로에 추가). arda-raset이
새로 갖고 있는 코드는 이 세 저장소를 엮어 하나의 프로세스로 띄우고, 원래
UDP 4~5개 채널이 하던 일을 대신하는 `raset/bus.py`의 큐/공유 상태뿐이다:

```
arda-radar --(낙하 좌표, coord_q)--------------> arda-servo
arda-radar --(관찰 트리거, trigger_q)----------> thermal-camera
arda-radar <--(person 판정, verdict_q)---------- thermal-camera
thermal-camera --(추적 보정, thermal_pan_q)----> arda-servo
arda-radar --(관찰 중인 낙하 위치, pending_location)--> thermal-camera
thermal-camera --(열원 검출 알림, thermal_engaged)----> arda-radar
```

`arda_servo.controller.ServoController.step()`처럼 원본 프로젝트의 코드는
글자 하나 안 바뀐 채 그대로 재사용된다 — UDP 소켓 대신 큐를 감싼 어댑터를
넘겨줄 뿐이다.

## 시각화

레이더의 matplotlib 3D 플롯은 GUI 이벤트 루프가 메인 스레드에서만 안전해서
쓰지 않는다(헤드리스) — 필요하면 `arda-radar`를 단독 실행해서 확인할 것.
열화상은 `--show-thermal`을 주면 관찰(dwell) 중에 컬러맵 창을 로컬
디스플레이에 띄운다(`DISPLAY` 필요). 그리고 `site.report_url`이 설정돼
있으면, 열화상이 낙하 후보를 관찰하는 동안 매 프레임 그 이미지를 함께
실어 실시간으로 웹에도 스트리밍한다(전송 포맷은 `arda-radar`의
`send_fall_report()` 참고). `report_url`이 비어 있으면(기본값) 이 전송은
발생하지 않는다.

## 사전 준비

```bash
cd arda-raset
uv sync                # 기본 (threshold 백엔드)
uv sync --extra yolo    # --yolo 쓰려면 추가 (Jetson CUDA torch는 jetson-ai-lab
                         # 인덱스에서 자동으로 받아옴 — requirements-yolo-jetson.txt와
                         # 동일 버전. 일반 PyPI torch는 이 보드에서 CUDA를 못 잡는다.)
```

`--yolo` 사용 시 CUDA가 여전히 안 잡히면 `thermal-camera/fix_cuda_cudss.sh`도
참고할 것.

세 프로젝트는 기본적으로 이 파일과 같은 부모 디렉터리(`ARDA-2026/`) 아래
형제 디렉터리로 있다고 가정한다. 다른 위치에 있으면 `ARDA_RADAR_DIR` /
`ARDA_SERVO_DIR` / `ARDA_THERMAL_DIR` 환경변수로 지정할 수 있다.

설정 파일(사이트 GPS, 서보 GPIO 핀, 카메라 geometry, `report_url` 등)은
arda-raset에 복제하지 않고 각 원본 저장소의 `config/settings.yaml`을 그대로
읽는다 — 단일 소스 유지. 다른 경로를 쓰려면 `--radar-settings` /
`--radar-profile` / `--servo-config`로 지정.

## 사용법

```bash
uv run python main.py                                # 전부 실제 하드웨어 (열화상은 threshold 판정)
uv run python main.py --yolo                          # 열화상 판정을 커스텀 YOLO 모델로
uv run python main.py --show-thermal                  # 열화상 컬러맵 창을 로컬에 표시
uv run python main.py --simulate-servo                # 서보만 시뮬레이션
uv run python main.py --simulate-thermal               # 열화상만 시뮬레이션 (센서 없이 임의 프레임)
uv run python main.py --no-radar                       # 레이더 강제 생략
uv run python main.py --simulate-servo --simulate-thermal --no-radar   # 하드웨어 전혀 없이 기동 확인용
```

레이더 USB(`/dev/ttyUSB0`, `/dev/ttyUSB1`)가 없으면 레이더만 자동으로
건너뛴다. 열화상 센서(MLX90640)가 없으면(`--simulate-thermal` 아닌 한)
열화상만 자동으로 건너뛴다 — 이 경우 레이더는 열화상 게이트 없이(즉시 GPS
로그) 동작한다. 서보는 항상 뜬다 — GPIO가 없으면 자동으로 시뮬레이션
모드가 된다.

로그는 `.logs/arda-raset.log` + 표준출력에 세 서브시스템이 섞여서 남는다.
Ctrl+C로 전체 종료(스레드 조인 최대 2초 대기 후 강제 종료 — 전부 daemon
스레드라 프로세스 자체는 즉시 끝난다).

## CLI 옵션

| 플래그 | 의미 | 기본값 |
|---|---|---|
| `--simulate-servo` | 서보 GPIO 없이 각도 계산만 | off |
| `--simulate-thermal` | 열화상 센서 없이 임의 프레임 | off |
| `--no-radar` | 레이더 강제 생략 | off |
| `--yolo` | 열화상 판정을 YOLO 백엔드로 | off (threshold) |
| `--show-thermal` | 열화상 컬러맵 창을 로컬 디스플레이에 표시 (`DISPLAY` 필요) | off |
| `--model-path` / `--confidence-threshold` / `--device` | YOLO 전용 | thermal-camera 기본값과 동일 |
| `--dwell-seconds` / `--required-consecutive` / `--settle-offset` | 열화상 관찰 파라미터 | 10.0 / 3 / 0.15 |
| `--thermal-pending-timeout` | 레이더가 열화상 판정을 기다리는 최대 시간(초) | 10.0 |
| `--radar-cli-port` / `--radar-data-port` | 레이더 시리얼 포트 | /dev/ttyUSB0 / ttyUSB1 |
| `--radar-settings` / `--radar-profile` / `--servo-config` | 원본 저장소 설정 파일 경로 | 형제 디렉터리 기준 |
