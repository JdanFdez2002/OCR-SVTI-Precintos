"""ISAPI serializado: paradas persistentes y prioridad manual por generación."""
import threading
from collections import deque

from PySide6.QtCore import QThread, Signal

STOP = dict(pan=0, tilt=0, zoom=0, focus=0, iris=0)


class PtzWorker(QThread):
    error = Signal(str)
    focus_status = Signal(str)
    configuration_status = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._evento = threading.Event()
        self._barriers = deque()
        self._comando = None
        self._activo = True
        self._service = self._ip = None
        self._generation = self._intent = 0
        self._zoom_active = self._manual = False
        self._auto = None
        self._motion_kind = ()
        self._initialized_generation = None
        self._last_was_stop = False
        self._stop_faults = set()
        self._settings = None
        self._cleanup_threads = []

    def configurar(self, service, ip):
        with self._lock:
            if self._service is not None:
                self._barriers.append((self._service, self._ip, True, False))
            self._generation += 1
            self._intent += 1
            self._service, self._ip = service, ip
            self._comando = None
            self._zoom_active = self._manual = False
            self._motion_kind = ()
            self._last_was_stop = False
            self._settings = None
            self._auto = (self._generation, self._intent) if service else None
        self._evento.set()

    def enviar(self, pan=0, tilt=0, zoom=0, focus=0, iris=0):
        command = dict(pan=pan, tilt=tilt, zoom=zoom, focus=focus, iris=iris)
        with self._lock:
            if not self._activo or self._service is None:
                return
            stop = not any(command.values())
            if stop and self._last_was_stop:
                return
            self._last_was_stop = stop
            self._intent += 1
            self._auto = None
            if focus:
                self._manual = True
                self._zoom_active = False
            if not any(command.values()):
                barrier = (self._service, self._ip, False, self._zoom_active)
                if not self._barriers or self._barriers[-1] != barrier:
                    self._barriers.append(barrier)
                self._comando = None
                self._zoom_active = False
                self._motion_kind = ()
            else:
                kind = tuple(k for k,v in command.items() if v)
                if (self._motion_kind and self._motion_kind != kind) or id(self._service) in self._stop_faults:
                    self._barriers.append((self._service, self._ip, False, False))
                self._motion_kind = kind
                if zoom:
                    self._zoom_active = True
                    self._manual = False
                self._comando = (self._generation, self._intent, command)
        self._evento.set()

    def cancelar_automatismos(self):
        with self._lock:
            self._auto = None
            self._intent += 1
            self._zoom_active = False
            self._manual = True
            self._last_was_stop = False
        self.enviar()

    def cambiar_dia_noche(self, modo):
        with self._lock:
            self._settings = (self._generation, modo)
        self._evento.set()

    def detener_hilo(self):
        self.configurar(None, None)
        with self._lock:
            self._activo = False
        self._evento.set()

    def _report(self, service, ok):
        if not ok:
            self.error.emit(service.ultimo_error or 'La cámara no confirmó la orden ISAPI')

    def run(self):
        while True:
            self._evento.wait()
            self._evento.clear()
            while True:
                with self._lock:
                    barrier = self._barriers.popleft() if self._barriers else None
                    command = self._comando if barrier is None else None
                    if command is not None:
                        self._comando = None
                    service, ip = self._service, self._ip
                    auto = self._auto if not barrier and not command else None
                    if auto:
                        self._auto = None
                    settings = self._settings if not barrier and not command and not auto and not self._motion_kind else None
                    if settings:
                        self._settings = None
                    active = self._activo
                if barrier:
                    target, address, retire, request_focus = barrier
                    ok = False
                    for _ in range(3):
                        ok = target.mover_camara(address, **STOP)
                        if ok:
                            break
                    self._report(target, ok)
                    with self._lock:
                        if not ok:
                            self._stop_faults.add(id(target))
                            self._last_was_stop = False
                            self._comando = self._auto = None
                        else:
                            self._stop_faults.discard(id(target))
                            if (request_focus and target is self._service and not self._manual
                                    and self._comando is None and not self._barriers):
                                self._auto = (self._generation, self._intent)
                    if retire:
                        cleanup = threading.Thread(target=target.cerrar, daemon=True)
                        self._cleanup_threads = [t for t in self._cleanup_threads if t.is_alive()]
                        self._cleanup_threads.append(cleanup)
                        cleanup.start()
                    continue
                if command:
                    with self._lock:
                        valid = (command[:2] == (self._generation, self._intent)
                                 and not self._barriers and self._activo)
                    if valid:
                        if command[2]['focus'] and hasattr(service, 'establecer_manual'):
                            def current():
                                with self._lock:
                                    return command[:2] == (self._generation, self._intent) and self._activo
                            manual_ok = service.establecer_manual(ip, current)
                            if not manual_ok:
                                focus_state = getattr(service, 'focus_capability_state', None)
                                focus_validation = getattr(service, 'focus_validation', None)
                                if focus_state in ('unknown', 'incompatible') and focus_validation:
                                    message = focus_validation
                                else:
                                    message = (
                                        service.ultimo_error
                                        or focus_validation
                                        or 'Modo MANUAL no confirmado; se conserva el mando manual original'
                                    )
                                self.focus_status.emit(message)
                            if not current():
                                continue
                        ok = service.mover_camara(ip, **command[2])
                        self._report(service, ok)
                        if not ok:
                            self.enviar()
                    continue
                if auto and service:
                    def valid():
                        with self._lock:
                            return (self._activo and auto == (self._generation, self._intent)
                                    and not self._manual and not self._barriers)
                    if valid() and self._initialized_generation != auto[0]:
                        self._initialized_generation = auto[0]
                        if hasattr(service, 'configurar_auto_iris'):
                            service.configurar_auto_iris(ip)
                    if valid() and hasattr(service, 'enfocar_nativo'):
                        status = service.enfocar_nativo(ip, valid)
                        if isinstance(status, str):
                            self.focus_status.emit(status)
                    continue
                if settings and service:
                    with self._lock:
                        valid_setting = (settings[0] == self._generation and not self._barriers and self._activo
                                         and self._comando is None and not self._motion_kind)
                    if valid_setting:
                        ok = service.configurar_modo_dia_noche(ip, settings[1])
                        self.configuration_status.emit('Modo cámara: '+settings[1] if ok else service.ultimo_error or 'Modo cámara rechazado')
                    continue
                if not active:
                    for cleanup in self._cleanup_threads:
                        cleanup.join()
                    return
                break
