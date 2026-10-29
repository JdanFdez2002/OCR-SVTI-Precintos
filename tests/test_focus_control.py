"""Todos los servicios y respuestas son simulados. Sin movimiento físico."""
import threading
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import Mock

from modulos.camera_service import CameraService
from modulos.ptz_worker import PtzWorker


class FocusServiceTests(unittest.TestCase):
    @staticmethod
    def xml_response(text, status_code=200):
        response = Mock(status_code=status_code, text=text)
        response.raise_for_status.return_value = None
        return response

    def make_service(self, modes='AUTO,MANUAL,SEMIAUTOMATIC'):
        service = CameraService('fixture','fixture')
        service.session = Mock()
        service.session.get.side_effect = [
            self.xml_response(f'<FocusConfiguration><focusStyle opt="{modes}"/></FocusConfiguration>'),
            self.xml_response('<FocusConfiguration><focusStyle>SEMIAUTOMATIC</focusStyle><focusLimited>1000</focusLimited></FocusConfiguration>')]
        service.session.put.return_value = Mock(status_code=200,text='<ResponseStatus><statusCode>1</statusCode></ResponseStatus>')
        return service

    def test_native_modes_require_advertised_capabilities_and_preserve_configuration(self):
        service = self.make_service()
        service.enfocar_nativo('fixture')
        self.assertEqual(service.session.get.call_count,2)
        xml = ET.fromstring(service.session.put.call_args.kwargs['data'])
        self.assertEqual(xml.findtext('focusStyle'),'AUTO')
        self.assertEqual(xml.findtext('focusLimited'),'1000')
        self.assertTrue(service.establecer_manual('fixture'))
        xml = ET.fromstring(service.session.put.call_args.kwargs['data'])
        self.assertEqual(xml.findtext('focusStyle'),'MANUAL')
        self.assertEqual(service.session.get.call_count,2)

    def test_unadvertised_modes_never_issue_automatic_put(self):
        service = self.make_service('MANUAL')
        status = service.enfocar_nativo('fixture')
        self.assertEqual(service.focus_capability_state, 'incompatible')
        self.assertIn('focusStyle@opt=MANUAL', status)
        service.session.put.assert_not_called()

    def test_capabilities_403_is_unknown_and_reports_real_error_without_put(self):
        service = CameraService('fixture', 'fixture')
        service.session = Mock()
        service.session.get.side_effect = [
            self.xml_response(
                '<ResponseStatus><statusCode>4</statusCode>'
                '<statusString>Invalid Operation</statusString>'
                '<subStatusCode>invalidOperation</subStatusCode>'
                '<description>unsupported capabilities endpoint</description>'
                '</ResponseStatus>',
                status_code=403,
            ),
            self.xml_response(
                '<FocusConfiguration><focusStyle>MANUAL</focusStyle>'
                '<focusLimited>150</focusLimited></FocusConfiguration>'
            ),
        ]

        status = service.enfocar_nativo('fixture')

        self.assertEqual(service.focus_capability_state, 'unknown')
        self.assertIn('HTTP 403', status)
        self.assertIn('invalidOperation', status)
        self.assertIn('focusStyle actual=MANUAL', status)
        self.assertIn('no se envía PUT automático', status)
        self.assertEqual(service.session.get.call_count, 2)
        service.session.put.assert_not_called()

    def test_manual_order_during_capability_get_cancels_pending_auto(self):
        service = self.make_service()
        valid = True
        original = service.session.get.side_effect
        def get(*args,**kwargs):
            nonlocal valid
            valid = False
            return original[0]
        service.session.get.side_effect = get
        service.enfocar_nativo('fixture',lambda:valid)
        service.session.put.assert_not_called()

    def test_http_rejected_auto_is_not_success(self):
        service = self.make_service()
        service.session.put.return_value = Mock(status_code=403,text='')
        self.assertFalse(service._focus_mode('fixture','AUTO',lambda:True))


class SimulatedCamera:
    ultimo_error = None
    def __init__(self):
        self.calls = []
        self.ready = threading.Event()
        self.auto_requested = threading.Event()
        self.manual_requested = threading.Event()
        self.block_stop = None
    def mover_camara(self,ip,**cmd):
        self.calls.append(('move',ip,cmd))
        if not any(cmd.values()) and self.block_stop:
            self.block_stop[0].set()
            self.block_stop[1].wait(2)
        self.ready.set()
        return True
    def enfocar_nativo(self,ip,valid):
        if valid():
            self.calls.append(('auto',ip))
            self.auto_requested.set()
            self.ready.set()
        return 'simulado'
    def establecer_manual(self,ip,valid):
        if valid():
            self.calls.append(('manual',ip))
            self.manual_requested.set()
        return True
    def configurar_auto_iris(self,ip):
        return True
    def cerrar(self):
        self.calls.append(('close',))


class FocusPriorityTests(unittest.TestCase):
    def run_worker(self,prepare):
        service,worker = SimulatedCamera(),PtzWorker()
        worker.configurar(service,'fixture')
        prepare(worker,service)
        worker.start()
        self.addCleanup(lambda: (worker.detener_hilo(),worker.wait(3000)))
        return worker,service

    def test_pending_startup_auto_is_cancelled_by_manual_focus(self):
        worker,service = self.run_worker(lambda w,s:w.enviar(focus=5))
        self.assertTrue(service.ready.wait(2))
        self.assertFalse(any(c[0]=='auto' for c in service.calls))
        self.assertEqual(service.calls[0][0],'manual')

    def test_zoom_stop_triggers_one_auto_with_repeated_stops(self):
        def prepare(w,s):
            w.enviar(zoom=20)
            w.enviar()
            w.enviar()
        worker,service = self.run_worker(prepare)
        self.assertTrue(service.auto_requested.wait(2))
        self.assertEqual(sum(c[0]=='auto' for c in service.calls),1)

    def test_manual_during_zoom_stop_invalidates_old_autofocus(self):
        entered,released = threading.Event(),threading.Event()
        def prepare(w,s):
            s.block_stop = (entered,released)
            w.enviar(zoom=20)
            w.enviar()
        worker,service = self.run_worker(prepare)
        self.assertTrue(entered.wait(2))
        worker.enviar(focus=-5)
        released.set()
        self.assertTrue(service.manual_requested.wait(2))
        self.assertFalse(any(c[0]=='auto' for c in service.calls))
        self.assertTrue(any(c[0]=='manual' for c in service.calls))

    def test_lost_video_cancels_zoom_recovery(self):
        def prepare(w,s):
            w.enviar(zoom=20)
            w.cancelar_automatismos()
        worker,service = self.run_worker(prepare)
        self.assertTrue(service.ready.wait(2))
        import time
        time.sleep(.03)
        self.assertFalse(any(c[0]=='auto' for c in service.calls))


if __name__=='__main__':
    unittest.main()
