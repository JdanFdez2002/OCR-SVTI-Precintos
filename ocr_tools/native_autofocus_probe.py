"""Prueba física controlada del autofocus nativo Hikvision.

El módulo sólo trabaja con ``/ISAPI/Image/channels/1/focusConfiguration``.
No contiene movimientos PTZ ni comandos de zoom. Antes de cualquier PUT pide
una autorización explícita que cubre tanto la solicitud AUTO como la
restauración del XML original.

Ejecutar desde la raíz del proyecto::

    .\\venv\\Scripts\\python.exe -m ocr_tools.native_autofocus_probe

La prueba real no debe ejecutarse desde una automatización. Las pruebas
automatizadas inyectan una sesión simulada y no abren conexiones de red.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable, Optional

import requests
from requests.auth import HTTPDigestAuth

from modulos.settings_store import load_camera_settings, settings_completos


FOCUS_CONFIGURATION_PATH = "/ISAPI/Image/channels/1/focusConfiguration"

_CREDENTIAL_URL_RE = re.compile(r"(?i)(https?://)[^/\s:@]+:[^/\s@]+@")
_SENSITIVE_NAME_RE = re.compile(
    r"(?i)(password|passwd|username|user|credential|secret|token|"
    r"authorization|sessionId|accessKey|apiKey|serialNumber|serial|"
    r"macAddress|deviceId|ipAddress|ipv4Address|ipv6Address)"
)
_SENSITIVE_TAG_RE = re.compile(
    r"(?is)(<(?P<open>[A-Za-z_][\w.-]*(?::[\w.-]+)?)\s*"
    r"(?:password|passwd|username|user|credential|secret|token|"
    r"authorization|sessionId|accessKey|apiKey)[^>]*>)"
    r".*?(</(?P=open)>)"
)


@dataclass(frozen=True)
class HttpResult:
    method: str
    path: str
    status_code: Optional[int]
    body: str
    content_type: Optional[str]
    error: Optional[str]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _sanitize_text(value: str) -> str:
    value = _CREDENTIAL_URL_RE.sub(r"\1[REDACTED]@", value)
    return _SENSITIVE_TAG_RE.sub(r"\1[REDACTED]\2", value)


def _sanitize_xml(value: str) -> str:
    try:
        root = ET.fromstring(value)
    except ET.ParseError:
        return _sanitize_text(value)

    for element in root.iter():
        if _SENSITIVE_NAME_RE.search(_local_name(element.tag)):
            element.text = "[REDACTED]"
            for child in element:
                child.clear()
        for attribute in list(element.attrib):
            if _SENSITIVE_NAME_RE.search(attribute):
                element.set(attribute, "[REDACTED]")
    return _sanitize_text(ET.tostring(root, encoding="unicode"))


class ReadOnlyAndFocusIsapiClient:
    """Client limited to GET/PUT of the focus configuration endpoint."""

    def __init__(
        self,
        ip: str,
        user: str,
        password: str,
        timeout: float = 3.0,
        session=None,
    ):
        self._url = f"http://{ip.strip()}{FOCUS_CONFIGURATION_PATH}"
        self._timeout = timeout
        if session is None:
            self._session = requests.Session()
            self._session.auth = HTTPDigestAuth(user, password)
            self._session.headers.update({"Accept": "application/xml"})
        else:
            self._session = session

    def close(self) -> None:
        self._session.close()

    def get_focus_configuration(self) -> HttpResult:
        try:
            response = self._session.get(self._url, timeout=self._timeout)
            return HttpResult(
                method="GET",
                path=FOCUS_CONFIGURATION_PATH,
                status_code=response.status_code,
                body=response.text,
                content_type=response.headers.get("Content-Type"),
                error=None,
            )
        except Exception as exc:
            return HttpResult(
                method="GET",
                path=FOCUS_CONFIGURATION_PATH,
                status_code=None,
                body="",
                content_type=None,
                error=_sanitize_text(f"{type(exc).__name__}: {exc}"),
            )

    def put_focus_configuration(self, xml: str) -> HttpResult:
        try:
            response = self._session.put(
                self._url,
                data=xml,
                timeout=self._timeout,
                headers={"Content-Type": "application/xml", "Accept": "application/xml"},
            )
            return HttpResult(
                method="PUT",
                path=FOCUS_CONFIGURATION_PATH,
                status_code=response.status_code,
                body=response.text,
                content_type=response.headers.get("Content-Type"),
                error=None,
            )
        except Exception as exc:
            return HttpResult(
                method="PUT",
                path=FOCUS_CONFIGURATION_PATH,
                status_code=None,
                body="",
                content_type=None,
                error=_sanitize_text(f"{type(exc).__name__}: {exc}"),
            )


def _print_result(result: HttpResult, output: Callable[[str], None]) -> None:
    output(f"\n{result.method} {result.path}")
    if result.status_code is None:
        output("HTTP: sin respuesta")
    else:
        output(
            f"HTTP: {result.status_code} | "
            f"Content-Type: {result.content_type or 'no informado'}"
        )
    if result.error:
        output(f"Error: {result.error}")
    if result.body:
        output("Respuesta XML/texto sanitizada:")
        output(_sanitize_xml(result.body))
    else:
        output("Respuesta: <vacía>")


def _parse_focus_configuration(xml: str) -> tuple[ET.Element, str]:
    root = ET.fromstring(xml)
    if _local_name(root.tag) != "FocusConfiguration":
        raise ValueError("La respuesta no tiene raíz FocusConfiguration")
    try:
        style = next(e for e in root.iter() if _local_name(e.tag) == "focusStyle")
    except StopIteration as exc:
        raise ValueError("La respuesta no contiene focusStyle") from exc
    value = (style.text or "").strip()
    if not value:
        raise ValueError("focusStyle está vacío")
    return root, value


def _set_focus_style(root: ET.Element, value: str) -> None:
    try:
        style = next(e for e in root.iter() if _local_name(e.tag) == "focusStyle")
    except StopIteration as exc:
        raise ValueError("El XML no contiene focusStyle") from exc
    style.text = value


def _put_response_confirmed(result: HttpResult) -> tuple[bool, str]:
    if result.status_code is None:
        return False, result.error or "sin respuesta HTTP"
    if not 200 <= result.status_code < 300:
        return False, f"HTTP {result.status_code} rechazó la operación"
    if not result.body.strip():
        return False, "HTTP 2xx sin respuesta ISAPI confirmatoria"
    try:
        root = ET.fromstring(result.body)
    except ET.ParseError:
        return False, "HTTP 2xx con respuesta no XML; cambio no confirmado"
    if _local_name(root.tag) != "ResponseStatus":
        return False, "HTTP 2xx sin raíz ResponseStatus; cambio no confirmado"
    codes = [
        (element.text or "").strip()
        for element in root.iter()
        if _local_name(element.tag) == "statusCode"
    ]
    if not codes:
        return False, "ResponseStatus sin statusCode; cambio no confirmado"
    if any(code != "1" for code in codes):
        return False, "Respuesta ISAPI rechazó el cambio: statusCode=" + str(codes)
    return True, "Respuesta ISAPI confirmatoria; falta verificar mediante GET"


class NativeAutofocusProbe:
    """Workflow interactivo, inyectable para pruebas sin cámara real."""

    def __init__(
        self,
        client: ReadOnlyAndFocusIsapiClient,
        input_fn: Callable[[str], str] = input,
        output: Callable[[str], None] = print,
    ):
        self.client = client
        self.input = input_fn
        self.output = output

    def _read_style(self, label: str) -> Optional[str]:
        self.output(f"\n{label}")
        result = self.client.get_focus_configuration()
        _print_result(result, self.output)
        if result.status_code is None or not 200 <= result.status_code < 300:
            self.output("No se pudo leer focusStyle mediante GET.")
            return None
        try:
            _, style = _parse_focus_configuration(result.body)
        except (ET.ParseError, ValueError) as exc:
            self.output(f"No se pudo interpretar focusStyle: {exc}")
            return None
        self.output(f"focusStyle leído: {style}")
        return style

    def _authorize_writes(self) -> bool:
        self.output(
            "\nAUTORIZACIÓN REQUERIDA: se ejecutarán como máximo dos PUT "
            "sobre focusConfiguration: AUTO y la restauración del XML original."
        )
        self.output("No se enviará ningún movimiento PTZ ni zoom automático.")
        try:
            answer = self.input("Escribe AUTORIZAR para continuar: ").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer == "AUTORIZAR"

    def _restore(self, original_xml: str, original_style: str) -> bool:
        self.output(f"\nRestaurando focusStyle original: {original_style}")
        result = self.client.put_focus_configuration(original_xml)
        _print_result(result, self.output)
        confirmed, detail = _put_response_confirmed(result)
        self.output(f"Confirmación del PUT de restauración: {detail}")
        restored_style = self._read_style("GET de verificación de restauración")
        if restored_style != original_style:
            self.output(
                "ADVERTENCIA: no se pudo verificar la restauración. "
                f"Esperado={original_style}; leído={restored_style or 'sin lectura'}."
            )
            return False
        if not confirmed:
            self.output(
                "Restauración verificada por GET, aunque la respuesta del PUT "
                "no fue confirmatoria."
            )
        else:
            self.output("Restauración verificada correctamente por GET.")
        return True

    def run(self) -> int:
        initial_result = self.client.get_focus_configuration()
        _print_result(initial_result, self.output)
        if initial_result.status_code is None or not 200 <= initial_result.status_code < 300:
            self.output("Diagnóstico detenido: no se pudo leer la configuración inicial.")
            return 1

        try:
            initial_root, original_style = _parse_focus_configuration(initial_result.body)
        except (ET.ParseError, ValueError) as exc:
            self.output(f"Diagnóstico detenido: XML inicial no válido: {exc}")
            return 1

        original_xml = initial_result.body
        self.output(f"focusStyle original guardado: {original_style}")

        if original_style == "AUTO":
            self.output(
                "La cámara ya está en AUTO; no se ejecuta un PUT redundante. "
                "Puede observar el enfoque después de hacer zoom manualmente."
            )
            self.input("Pulsa Enter al terminar la observación manual: ")
            final_style = self._read_style("GET final sin cambios")
            return 0 if final_style == original_style else 1

        if not self._authorize_writes():
            self.output("Prueba no autorizada: no se ejecutó ningún PUT.")
            return 0

        result_code = 0
        try:
            auto_root = ET.fromstring(initial_result.body)
            _set_focus_style(auto_root, "AUTO")
            auto_xml = ET.tostring(auto_root, encoding="unicode")

            self.output("\nSolicitando focusStyle=AUTO mediante PUT...")
            auto_result = self.client.put_focus_configuration(auto_xml)
            _print_result(auto_result, self.output)
            confirmed, detail = _put_response_confirmed(auto_result)
            self.output(f"Confirmación del PUT AUTO: {detail}")
            if not confirmed:
                self.output(
                    "HTTP 200/2xx no se considera suficiente; la aplicación "
                    "se decidirá con el GET posterior."
                )

            observed_style = self._read_style("GET de verificación después de AUTO")
            if observed_style != "AUTO":
                self.output(
                    "AUTO no quedó verificado por GET; no se declara autofocus activo."
                )
                result_code = 1
            else:
                self.output(
                    "AUTO verificado por GET. Ahora realice el zoom manualmente "
                    "y observe si la cámara reenfoca."
                )
                self.input("Pulse Enter cuando termine la observación física: ")
        except Exception as exc:
            self.output(f"Error durante la prueba AUTO: {type(exc).__name__}: {exc}")
            result_code = 1
        finally:
            if not self._restore(original_xml, original_style):
                result_code = 2
        return result_code


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Timeout de cada consulta HTTP en segundos (por defecto: 3).",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout debe ser mayor que cero")

    cfg = load_camera_settings()
    if not settings_completos(cfg):
        print(
            "Configuración incompleta: se requieren IP, usuario, contraseña y canal.",
            file=sys.stderr,
        )
        return 2

    client = ReadOnlyAndFocusIsapiClient(
        cfg.ip,
        cfg.usuario,
        cfg.password,
        timeout=args.timeout,
    )
    try:
        return NativeAutofocusProbe(client).run()
    except KeyboardInterrupt:
        print("\nPrueba interrumpida; se intentó ejecutar la restauración si correspondía.")
        return 130
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
