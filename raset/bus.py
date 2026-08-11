"""arda-radar / arda-servo / thermal-camera 사이의 UDP 4채널을 대체하는
인메모리 큐 + 공유 상태.

큐에는 JSON 대신 각 프로젝트의 원본 dataclass를 그대로 담는다 — 같은
프로세스 안이라 직렬화가 불필요하고, ServoController처럼 그 타입의 필드에
직접 접근하는 코드를 손대지 않고 그대로 재사용할 수 있다. 호출부에서는 항상
키워드 인자로만 생성할 것 (Coord는 x,y,z,fall,ts,confidence 순서라 위치
인자로 넣으면 실수하기 쉽다).
"""

import queue
import threading
import time
from dataclasses import dataclass

from arda_servo.receiver import Coord  # noqa: F401 — 재export, radar_worker/servo_worker에서 씀
from arda_servo.thermal_receiver import ThermalPan  # noqa: F401
from arda.utils.thermal_receiver import ThermalVerdict  # noqa: F401


class LatestQueue:
    """maxsize=1로 항상 최신 값만 유지하는 큐 — put()이 기존 값을 밀어내고
    교체한다. UDP는 커널 버퍼가 사실상 backpressure 역할을 했지만 일반
    queue.Queue()는 무제한 적재되어, 소비자가 잠깐 멈추면 오래된 값이 나중에
    재생되는 문제가 생길 수 있다 — 0.5초마다 오는 열화상 추적보정처럼 "최신
    값만 의미 있는" 채널에 쓴다."""

    def __init__(self):
        self._q: queue.Queue = queue.Queue(maxsize=1)

    def put(self, item) -> None:
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        self._q.put_nowait(item)

    def get(self, timeout: float | None = None):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


class BoundedQueue:
    """이산 이벤트(낙하 확정, 관찰 시작, 판정 완료)용 평범한 FIFO. 상한을 둬서
    소비자가 죽었거나 멈췄을 때 무한정 쌓이지 않게 한다 — 가득 차면 가장
    오래된 항목을 버리고 새 항목을 넣는다(최신 이벤트가 항상 더 중요함)."""

    def __init__(self, maxsize: int = 32):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)

    def put(self, item) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(item)

    def get(self, timeout: float | None = None):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


class QueueReceiver:
    """ServoController가 기대하는 `.recv()`(인자 없이 호출)/`.close()`
    인터페이스에 맞춘 어댑터. 실제 폴링 타임아웃은 생성 시점에 고정해서,
    원래 UDP 리시버들의 socket timeout과 동일한 페이싱을 재현한다(좌표
    0.5s / 열화상보정 0.01s) — queue.Queue.get(timeout=X)도 socket timeout과
    마찬가지로 진짜 OS 레벨 sleep이라 busy-poll이 아니다."""

    def __init__(self, q, timeout: float):
        self._q = q
        self._timeout = timeout

    def recv(self):
        return self._q.get(timeout=self._timeout)

    def close(self) -> None:
        pass


@dataclass
class PendingLocation:
    lat: float
    lon: float
    ts: float


class LocationHolder:
    """"현재 관찰 중인 낙하의 위치" 하나만 여러 스레드가 읽고 쓰는 공유 상태
    — 이벤트 큐가 아니라 최신 값 하나를 계속 덮어쓰는 것이라 Lock으로 감싼
    홀더로 따로 둔다. radar_worker가 열화상 트리거를 보낼 때 set(), 관찰이
    끝나면(판정 수신·타임아웃) clear() — thermal_worker는 매 프레임 get()해서
    값이 있을 때만(=관찰 중일 때만) 열화상 스트리밍 리포트를 보낸다."""

    def __init__(self):
        self._lock = threading.Lock()
        self._value: PendingLocation | None = None

    def set(self, lat: float, lon: float) -> None:
        with self._lock:
            self._value = PendingLocation(lat=lat, lon=lon, ts=time.time())

    def clear(self) -> None:
        with self._lock:
            self._value = None

    def get(self) -> PendingLocation | None:
        with self._lock:
            return self._value


class Bus:
    """세 워커 스레드가 공유하는 채널 모음 — 원래 UDP 4개 링크 + 위치 공유 상태."""

    def __init__(self):
        self.coord_q = BoundedQueue(maxsize=32)  # radar -> servo (Coord)
        self.trigger_q = BoundedQueue(maxsize=32)  # radar -> thermal (float ts)
        self.verdict_q = BoundedQueue(maxsize=32)  # thermal -> radar (ThermalVerdict)
        self.thermal_pan_q = LatestQueue()  # thermal -> servo (ThermalPan)
        self.pending_location = LocationHolder()  # radar -> thermal (관찰 중인 낙하 위치)
        # thermal -> radar. 지금 진행 중인 관찰에서 열화상이 발열 영역을 한 번이라도
        # 검출했는지(=열원을 붙잡아 추적을 시작했는지) — arda_servo.controller.
        # ServoController의 _thermal_engaged와 같은 원칙을 radar_worker의
        # confidence 기반 선점 판단에도 적용하기 위한 것. thermal_worker가 관찰
        # 시작 시 clear(), 발열 영역을 검출하면 set()한다.
        self.thermal_engaged = threading.Event()
