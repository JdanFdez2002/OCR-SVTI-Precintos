"""
Módulo de gestión de cámara IP (Video en vivo y Control PTZ).

 Captura de video IRL
   - Se conecta al stream RTSP de la cámara usando OpenCV
   - Reduce el retraso (lag) al mínimo configurando el buffer en 1 fotograma
   - Reconecta automáticamente si se pierde la conexión o la señal de video

 Acceso seguro a imágenes (Thread-safe)
   - Guarda el último fotograma en segundo plano y permite consultarlo desde otros módulos sin congelar la app (usando Locks).

 Control de movimiento y lente (Protocolo ISAPI de HIKVISION):
   - Mueve la cámara (rotación horizontal/vertical y zoom).
   - Controla el enfoque manual y activa el ajuste automático del diafragma (Iris).
   - Maneja la autenticación y las peticiones HTTP seguras con la cámara.
"""

import threading
import time
import xml.etree.ElementTree as ET
import copy
from urllib.parse import quote

import cv2
import requests
from requests.auth import HTTPDigestAuth


class CameraService:
    def __init__(self, user: str, password: str):
        self.ultimo_frame = None
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self.conectado = False
        self.ultimo_error = None
        self.session = requests.Session()
        self.session.auth = HTTPDigestAuth(user, password)
        self._frame_at = 0.0
        self.frame_sequence = 0
        self.stream_epoch = 0
        self._reader = None
        self._focus_checked = False
        self._focus_config = None
        self._focus_modes = set()
        self.focus_validation = 'Pendiente de validar capacidades y funcionamiento físico'

    def _focus_capabilities(self, ip, valid):
        if self._focus_checked:
            return self._focus_config is not None
        base = f'http://{ip}/ISAPI/Image/channels/1/focusConfiguration'
        try:
            if not valid():
                return False
            response = self.session.get(base + '/capabilities', timeout=.5)
            response.raise_for_status()
            capabilities = ET.fromstring(response.text)
            if capabilities.tag.split('}')[-1] != 'FocusConfiguration':
                raise ValueError('La cámara no devuelve capacidades FocusConfiguration')
            style = next(e for e in capabilities.iter() if e.tag.split('}')[-1] == 'focusStyle')
            modes = set(style.get('opt', '').split(','))
            if not {'AUTO', 'MANUAL'} <= modes:
                raise ValueError('No anuncia AUTO y MANUAL; se conserva control manual original')
            if not valid():
                return False
            response = self.session.get(base, timeout=.5)
            response.raise_for_status()
            config = ET.fromstring(response.text)
            if config.tag.split('}')[-1] != 'FocusConfiguration':
                raise ValueError('Configuración de enfoque no válida')
            next(e for e in config.iter() if e.tag.split('}')[-1] == 'focusStyle')
            self._focus_config, self._focus_modes = config, modes
            self._focus_checked = True
            self.focus_validation = 'AUTO/MANUAL anunciados por ISAPI; efecto óptico pendiente de prueba física'
            return True
        except Exception:
            self._focus_checked = True
            self.focus_validation = 'Autofocus no validado por ISAPI; control manual disponible'
            return False

    def _focus_mode(self, ip, mode, valid):
        if not self._focus_capabilities(ip, valid) or not valid():
            return False
        config = copy.deepcopy(self._focus_config)
        style = next(e for e in config.iter() if e.tag.split('}')[-1] == 'focusStyle')
        style.text = mode
        if not valid():
            return False
        ok = self._put_xml(f'http://{ip}/ISAPI/Image/channels/1/focusConfiguration',
                           ET.tostring(config, encoding='unicode'))
        if ok:
            self._focus_config = config
        return ok

    def enfocar_nativo(self, ip, valid=lambda: True):
        ok = self._focus_mode(ip, 'AUTO', valid)
        if not valid():
            return 'Autofocus cancelado por una orden posterior'
        return ('Enfoque nativo AUTO solicitado; validar efecto físico' if ok else
                (self.ultimo_error or self.focus_validation))

    def establecer_manual(self, ip, valid=lambda: True):
        # Sólo se cambia el modo si la cámara lo anuncia. El movimiento manual
        # original permanece disponible si no ofrece FocusConfiguration.
        if not self._focus_capabilities(ip, valid):
            return False
        return self._focus_mode(ip, 'MANUAL', valid)

    def iniciar_lectura(self, url):
        self._reader = threading.Thread(target=self.lector_rtsp, args=(url,), daemon=True)
        self._reader.start()
        return self._reader

    def _invalidar_video(self, error=None):
        with self.lock:
            self.conectado = False
            self.ultimo_frame = None
            self._frame_at = 0.0
            self.stream_epoch += 1
        if error:
            self.ultimo_error = error

    @staticmethod
    def url_rtsp(user: str, password: str, ip: str, canal: str = "101", puerto: int = 554) -> str:
        u = quote(user, safe="")
        p = quote(password, safe="")
        return f"rtsp://{u}:{p}@{ip}:{puerto}/Streaming/Channels/{canal}"

    def detener_lectura(self):
        self._stop.set()
        self._invalidar_video()

    def cerrar(self):
        self.detener_lectura()
        if self._reader is not None and self._reader is not threading.current_thread():
            self._reader.join()
        self.session.close()

    @staticmethod
    def _abrir_rtsp(url):
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000,
        ])

    def lector_rtsp(self, rtsp_url: str):
        self._invalidar_video()
        while not self._stop.is_set():
            cap = None
            try:
                cap = self._abrir_rtsp(rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    self._invalidar_video("No se pudo abrir el stream RTSP")
                else:
                    while not self._stop.is_set():
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            self._invalidar_video("Se perdió la señal de video")
                            break
                        with self.lock:
                            if self._stop.is_set():
                                break
                            self.ultimo_frame = frame.copy()
                            self._frame_at = time.monotonic()
                            self.frame_sequence += 1
                            self.conectado = True
            except Exception:
                self._invalidar_video("Error de transporte RTSP")
            finally:
                if cap is not None:
                    cap.release()
            self._stop.wait(1)
        self._invalidar_video()

    def copiar_muestra(self, max_age_s=1.5, *, after_sequence=None):
        """Muestra reciente; frame=None indica secuencia ya vista, no señal perdida.

        La caducidad se comprueba antes de evitar la copia del duplicado.
        """
        with self.lock:
            if (self._stop.is_set() or not self.conectado or self.ultimo_frame is None
                    or time.monotonic() - self._frame_at > max_age_s):
                return None
            frame = None if self.frame_sequence == after_sequence else self.ultimo_frame.copy()
            return frame, self.frame_sequence, self._frame_at, self.stream_epoch

    def copiar_frame(self):
        sample = self.copiar_muestra()
        return sample[0] if sample is not None else None

    @staticmethod
    def capturar_frame_rtsp(rtsp_url: str, intentos: int = 3):
        cap = CameraService._abrir_rtsp(rtsp_url)
        try:
            if not cap.isOpened():
                return None
            for _ in range(intentos):
                ret, actual = cap.read()
                if ret and actual is not None:
                    return actual.copy()
            return None
        finally:
            cap.release()

    def _put_xml(self, url, xml, timeout=.5):
        try:
            response = self.session.put(url, data=xml, timeout=timeout,
                                        headers={"Content-Type": "application/xml"})
            if not 200 <= response.status_code < 300:
                raise ValueError(f"Rechazo HTTP {response.status_code}")
            if response.text.strip():
                root = ET.fromstring(response.text)
                codes = [e.text for e in root.iter() if e.tag.split('}')[-1] == "statusCode"]
                if root.tag.split('}')[-1] != 'ResponseStatus' or not codes:
                    raise ValueError('Respuesta ISAPI inesperada; operación no confirmada')
                if any((code or '').strip() != "1" for code in codes):
                    raise ValueError("Rechazo ISAPI: statusCode=" + str(codes))
            self.ultimo_error = None
            return True
        except Exception as exc:
            self.ultimo_error = str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: petición ISAPI fallida"
            return False

    def mover_camara(self, ip, pan=0, tilt=0, zoom=0, focus=0, iris=0) -> bool:
        detener = pan == 0 and tilt == 0 and zoom == 0 and focus == 0 and iris == 0
        outcomes, errors = [], []
        if pan != 0 or tilt != 0 or zoom != 0 or detener:
            url = f"http://{ip}/ISAPI/PTZCtrl/channels/1/continuous"
            xml = f"<PTZData><pan>{pan}</pan><tilt>{tilt}</tilt><zoom>{zoom}</zoom></PTZData>"
            outcomes.append(self._put_xml(url, xml))
            if not outcomes[-1]:
                errors.append(self.ultimo_error)

        if focus != 0 or detener:
            url = f"http://{ip}/ISAPI/System/Video/inputs/channels/1/focus"
            xml = f"<FocusData><focus>{focus}</focus></FocusData>"
            outcomes.append(self._put_xml(url, xml))
            if not outcomes[-1]:
                errors.append(self.ultimo_error)
        self.ultimo_error = "; ".join(errors) if errors else None
        return bool(outcomes) and all(outcomes)

    def configurar_auto_iris(self, ip) -> bool:
        url = f"http://{ip}/ISAPI/Image/channels/1/iris"
        xml = "<Iris><irisMode>auto</irisMode></Iris>"
        return self._put_xml(url, xml, timeout=1)

    def configurar_modo_dia_noche(self, ip, modo: str) -> bool:
        valores = {
            "dia": "day",
            "día": "day",
            "noche": "night",
            "auto": "auto",
        }
        valor = valores.get((modo or "").strip().lower())
        if valor is None:
            self.ultimo_error = f"Modo dia/noche no valido: {modo}"
            return False

        url = f"http://{ip}/ISAPI/Image/channels/1/ircutFilter"
        xml = (
            "<IrcutFilter>"
            f"<IrcutFilterType>{valor}</IrcutFilterType>"
            "</IrcutFilter>"
        )
        return self._put_xml(url, xml, timeout=2)

