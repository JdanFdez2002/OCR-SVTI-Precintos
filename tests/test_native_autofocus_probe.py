"""Pruebas simuladas del flujo de autofocus; nunca conectan a la cámara."""

import unittest
from collections import deque
from xml.etree import ElementTree as ET

from ocr_tools.native_autofocus_probe import (
    NativeAutofocusProbe,
    ReadOnlyAndFocusIsapiClient,
)


class FakeResponse:
    def __init__(self, status_code, text, content_type="application/xml"):
        self.status_code = status_code
        self.text = text
        self.headers = {"Content-Type": content_type}


class FakeSession:
    def __init__(self, get_responses, put_responses):
        self.get_responses = deque(get_responses)
        self.put_responses = deque(put_responses)
        self.get_calls = []
        self.put_calls = []

    def get(self, url, timeout):
        self.get_calls.append((url, timeout))
        return self.get_responses.popleft()

    def put(self, url, data, timeout, headers):
        self.put_calls.append((url, data, timeout, headers))
        return self.put_responses.popleft()

    def close(self):
        return None


def focus_xml(style, extra=""):
    return (
        "<FocusConfiguration>"
        f"<focusStyle>{style}</focusStyle>{extra}"
        "</FocusConfiguration>"
    )


def response_status(code="1"):
    return f"<ResponseStatus><statusCode>{code}</statusCode></ResponseStatus>"


class NativeAutofocusProbeTests(unittest.TestCase):
    def make_probe(self, session, answers):
        client = ReadOnlyAndFocusIsapiClient(
            "fixture",
            "fixture-user",
            "fixture-password",
            session=session,
        )
        output = []
        answer_queue = deque(answers)
        probe = NativeAutofocusProbe(
            client,
            input_fn=lambda _prompt: answer_queue.popleft(),
            output=output.append,
        )
        return client, probe, output

    def test_declined_authorization_never_executes_put(self):
        session = FakeSession(
            [FakeResponse(200, focus_xml("MANUAL"))],
            [],
        )
        client, probe, output = self.make_probe(session, ["NO"])
        try:
            code = probe.run()
        finally:
            client.close()

        self.assertEqual(code, 0)
        self.assertEqual(session.put_calls, [])
        self.assertIn("no se ejecutó ningún PUT", "\n".join(output))

    def test_http_200_without_body_is_verified_by_get_and_restored(self):
        session = FakeSession(
            [
                FakeResponse(200, focus_xml("MANUAL")),
                FakeResponse(200, focus_xml("AUTO")),
                FakeResponse(200, focus_xml("MANUAL")),
            ],
            [
                FakeResponse(200, ""),
                FakeResponse(200, response_status("1")),
            ],
        )
        client, probe, output = self.make_probe(session, ["AUTORIZAR", ""])
        try:
            code = probe.run()
        finally:
            client.close()

        self.assertEqual(code, 0)
        self.assertEqual(len(session.put_calls), 2)
        first_xml = ET.fromstring(session.put_calls[0][1])
        restored_xml = ET.fromstring(session.put_calls[1][1])
        self.assertEqual(first_xml.findtext("focusStyle"), "AUTO")
        self.assertEqual(restored_xml.findtext("focusStyle"), "MANUAL")
        self.assertIn("HTTP 2xx sin respuesta ISAPI confirmatoria", "\n".join(output))
        self.assertIn("Restauración verificada correctamente", "\n".join(output))

    def test_error_during_observation_still_restores_original_xml(self):
        session = FakeSession(
            [
                FakeResponse(200, focus_xml("MANUAL")),
                FakeResponse(200, focus_xml("AUTO")),
                FakeResponse(200, focus_xml("MANUAL")),
            ],
            [
                FakeResponse(200, response_status("1")),
                FakeResponse(200, response_status("1")),
            ],
        )
        client = ReadOnlyAndFocusIsapiClient("fixture", "u", "p", session=session)
        output = []
        prompts = deque(["AUTORIZAR"])

        def input_fn(_prompt):
            if prompts:
                return prompts.popleft()
            raise RuntimeError("observación simulada interrumpida")

        probe = NativeAutofocusProbe(client, input_fn=input_fn, output=output.append)
        try:
            code = probe.run()
        finally:
            client.close()

        self.assertEqual(code, 1)
        self.assertEqual(len(session.put_calls), 2)
        restored_xml = ET.fromstring(session.put_calls[-1][1])
        self.assertEqual(restored_xml.findtext("focusStyle"), "MANUAL")
        self.assertIn("Error durante la prueba AUTO", "\n".join(output))

    def test_failed_restoration_emits_clear_warning(self):
        session = FakeSession(
            [
                FakeResponse(200, focus_xml("MANUAL")),
                FakeResponse(200, focus_xml("AUTO")),
                FakeResponse(200, focus_xml("AUTO")),
            ],
            [
                FakeResponse(200, response_status("1")),
                FakeResponse(500, "<ResponseStatus><statusCode>4</statusCode></ResponseStatus>"),
            ],
        )
        client, probe, output = self.make_probe(session, ["AUTORIZAR", ""])
        try:
            code = probe.run()
        finally:
            client.close()

        self.assertEqual(code, 2)
        self.assertIn("ADVERTENCIA: no se pudo verificar la restauración", "\n".join(output))

    def test_sensitive_values_are_not_printed(self):
        initial = focus_xml("MANUAL", "<password>secret-value</password>")
        session = FakeSession([FakeResponse(200, initial)], [])
        client, probe, output = self.make_probe(session, ["NO"])
        try:
            probe.run()
        finally:
            client.close()

        self.assertNotIn("secret-value", "\n".join(output))


if __name__ == "__main__":
    unittest.main()
