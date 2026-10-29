"""Diagnóstico ISAPI de sólo lectura para modelo, firmware y enfoque Hikvision.

Este módulo usa la configuración de conexión existente, pero no importa
``CameraService``: toda petición de red se realiza exclusivamente mediante
``requests.Session.get``. No ejecuta PUT, POST, DELETE ni movimientos PTZ.

Ejecutar desde la raíz del proyecto::

    .\\venv\\Scripts\\python.exe -m ocr_tools.inspect_focus_isapi

Las respuestas XML se muestran sanitizadas para no exponer credenciales ni
identificadores sensibles de la cámara.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

import requests
from requests.auth import HTTPDigestAuth

from modulos.settings_store import load_camera_settings, settings_completos


DEVICE_INFO_PATH = "/ISAPI/System/deviceInfo"
FIRMWARE_VERSION_PATH = "/ISAPI/System/firmwareVersion"
FOCUS_CAPABILITIES_PATH = (
    "/ISAPI/Image/channels/1/focusConfiguration/capabilities"
)
FOCUS_CONFIGURATION_PATH = "/ISAPI/Image/channels/1/focusConfiguration"
MANUAL_FOCUS_CAPABILITIES_PATH = (
    "/ISAPI/System/Video/inputs/channels/1/focus/capabilities"
)
MANUAL_FOCUS_PATH = "/ISAPI/System/Video/inputs/channels/1/focus"

READ_ONLY_PATHS = (
    DEVICE_INFO_PATH,
    FIRMWARE_VERSION_PATH,
    FOCUS_CAPABILITIES_PATH,
    FOCUS_CONFIGURATION_PATH,
    MANUAL_FOCUS_CAPABILITIES_PATH,
    MANUAL_FOCUS_PATH,
)

_CREDENTIAL_URL_RE = re.compile(
    r"(?i)(https?://)[^/\s:@]+:[^/\s@]+@"
)
_SENSITIVE_NAME_RE = re.compile(
    r"(?i)(password|passwd|username|userName|user|credential|secret|token|"
    r"authorization|sessionId|accessKey|apiKey|serialNumber|serial|"
    r"macAddress|deviceId|ipAddress|ipv4Address|ipv6Address)"
)
_SENSITIVE_TAG_RE = re.compile(
    r"(?is)(<(?P<open>[A-Za-z_][\w.-]*(?::[\w.-]+)?)\s*"
    r"(?:password|passwd|username|user|credential|secret|token|"
    r"authorization|sessionId|accessKey|apiKey)[^>]*>)"
    r".*?(</(?P=open)>)"
)


@dataclass
class ProbeResult:
    path: str
    status_code: Optional[int]
    elapsed_ms: Optional[int]
    content_type: Optional[str]
    body: str
    xml_root: Optional[ET.Element]
    error: Optional[str]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _sanitize_text(value: str) -> str:
    """Remove credential-bearing URL fragments from text or errors."""

    value = _CREDENTIAL_URL_RE.sub(r"\1[REDACTED]@", value)
    return _SENSITIVE_TAG_RE.sub(r"\1[REDACTED]\2", value)


def _sanitize_xml(raw: str) -> tuple[str, Optional[ET.Element]]:
    """Return printable XML with sensitive values redacted when possible."""

    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return _sanitize_text(raw), None

    for element in root.iter():
        if _SENSITIVE_NAME_RE.search(_local_name(element.tag)):
            element.text = "[REDACTED]"
            for child in element:
                child.clear()
        for attribute in list(element.attrib):
            if _SENSITIVE_NAME_RE.search(attribute):
                element.set(attribute, "[REDACTED]")

    sanitized = ET.tostring(root, encoding="unicode")
    return _sanitize_text(sanitized), root


class ReadOnlyIsapiClient:
    """Small allow-listed ISAPI client whose only network verb is GET."""

    def __init__(self, ip: str, user: str, password: str, timeout: float):
        self._base_url = f"http://{ip.strip()}"
        self._timeout = timeout
        self._session = requests.Session()
        self._session.auth = HTTPDigestAuth(user, password)
        self._session.headers.update({"Accept": "application/xml"})

    def close(self) -> None:
        self._session.close()

    def get(self, path: str) -> ProbeResult:
        if path not in READ_ONLY_PATHS:
            raise ValueError(f"Ruta no permitida por el diagnóstico GET-only: {path}")

        started = time.perf_counter()
        try:
            response = self._session.get(
                self._base_url + path,
                timeout=self._timeout,
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            body, xml_root = _sanitize_xml(response.text)
            return ProbeResult(
                path=path,
                status_code=response.status_code,
                elapsed_ms=elapsed_ms,
                content_type=response.headers.get("Content-Type"),
                body=body,
                xml_root=xml_root,
                error=None,
            )
        except requests.RequestException as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            message = _sanitize_text(str(exc))
            return ProbeResult(
                path=path,
                status_code=None,
                elapsed_ms=elapsed_ms,
                content_type=None,
                body="",
                xml_root=None,
                error=f"{type(exc).__name__}: {message}",
            )


def _first_text(root: Optional[ET.Element], names: tuple[str, ...]) -> Optional[str]:
    if root is None:
        return None
    wanted = set(names)
    for element in root.iter():
        if _local_name(element.tag) in wanted and element.text:
            value = element.text.strip()
            if value:
                return value
    return None


def _focus_modes(root: Optional[ET.Element]) -> list[str]:
    if root is None:
        return []
    modes: list[str] = []
    for element in root.iter():
        if _local_name(element.tag) != "focusStyle":
            continue
        modes.extend(
            value.strip()
            for value in element.attrib.get("opt", "").split(",")
            if value.strip()
        )
        if element.text and element.text.strip():
            modes.append(f"actual={element.text.strip()}")
    return list(dict.fromkeys(modes))


def _print_result(result: ProbeResult) -> None:
    print(f"\nGET {result.path}")
    if result.status_code is None:
        print("HTTP: sin respuesta")
    else:
        content_type = result.content_type or "no informado"
        print(
            f"HTTP: {result.status_code} | "
            f"tiempo: {result.elapsed_ms} ms | "
            f"Content-Type: {content_type}"
        )
    if result.error:
        print(f"Error: {result.error}")
    if result.body:
        label = "XML (sanitizado)" if result.xml_root is not None else "Respuesta (sanitizada)"
        print(f"{label}:")
        print(result.body)


def _print_summary(results: list[ProbeResult]) -> None:
    by_path = {result.path: result for result in results}
    device_info = by_path.get(DEVICE_INFO_PATH)
    firmware_info = by_path.get(FIRMWARE_VERSION_PATH)
    focus_capabilities = by_path.get(FOCUS_CAPABILITIES_PATH)

    model = _first_text(device_info.xml_root if device_info else None, ("model",))
    firmware = _first_text(
        device_info.xml_root if device_info else None,
        ("firmwareVersion", "version"),
    )
    if firmware is None:
        firmware = _first_text(
            firmware_info.xml_root if firmware_info else None,
            ("firmwareVersion", "version"),
        )

    print("\nResumen extraído")
    print(f"Modelo: {model or 'no informado'}")
    print(f"Firmware: {firmware or 'no informado'}")
    modes = _focus_modes(
        focus_capabilities.xml_root if focus_capabilities else None
    )
    print(
        "Modos focusStyle anunciados/observados: "
        + (", ".join(modes) if modes else "no informado")
    )
    print(
        "Nota: un HTTP 404/405/501 o un timeout se informa como resultado del "
        "endpoint y no detiene las consultas restantes."
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Timeout por consulta GET en segundos (por defecto: 3).",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout debe ser mayor que cero")

    cfg = load_camera_settings()
    if not settings_completos(cfg):
        print(
            "Configuración incompleta: se requieren IP, usuario, contraseña y canal "
            "en la configuración existente.",
            file=sys.stderr,
        )
        return 2

    print("Diagnóstico Hikvision ISAPI: sólo GET; no se ejecutan PUT/POST/PTZ.")
    client = ReadOnlyIsapiClient(cfg.ip, cfg.usuario, cfg.password, args.timeout)
    results: list[ProbeResult] = []
    try:
        for path in READ_ONLY_PATHS:
            result = client.get(path)
            results.append(result)
            _print_result(result)
    finally:
        client.close()

    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
