"""Integración Qt offscreen sin red, vídeo ni modelos nativos."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QLineEdit

from modulos.excel_manifest import RegistroEsperado
from modulos.ocr_analysis import AnalysisResult, Candidate, Detection
from modulos.ocr_worker import OcrWorker
from modulos.ptz_worker import PtzWorker
from ui.main_window import MainWindow
from motor import MotorWindow
from modulos.ocr_reader import RecognitionResult
from PySide6.QtWidgets import QTableWidgetItem
from config import CANAL_RTSP_DEFAULT


class UiSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.patches = [patch.object(OcrWorker,'start'), patch.object(PtzWorker,'start'),
                        patch.object(MainWindow,'_inicio')]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.window = MainWindow()
        self.window._timer.stop()
        self.addCleanup(self.window.deleteLater)

    def test_lost_video_clears_picture_overlay_and_capture(self):
        w = self.window
        w._video.mostrar_frame(np.zeros((200,300,3),np.uint8))
        w._ocr_listo = True
        w._video_epoch = 1
        w._service = Mock(ultimo_error='Sin señal')
        w._service.copiar_muestra.return_value = None
        w._tick_video()
        self.assertFalse(w._video.tiene_imagen())
        self.assertFalse(w._btn_capturar.isEnabled())
        self.assertIsNone(w._video_epoch)

    def test_manual_capture_passes_entire_photo_without_manifest(self):
        w = self.window
        w._generation,w._video_epoch = 3,2
        w._service = Mock()
        frame = np.zeros((180,320,3),np.uint8)
        frame[165,10] = 255  # Evidencia fuera de la retícula central.
        with tempfile.TemporaryDirectory() as tmp,patch('ui.main_window.asegurar_carpeta_capturas',return_value=Path(tmp)),patch.object(w._ocr,'pedir_ocr') as read:
            w._procesar_captura_alta(3,dict(frame=frame,stream_epoch=2))
            np.testing.assert_array_equal(read.call_args.args[0],frame)
            self.assertEqual(read.call_args.kwargs['input_mode'],'full_photo')
            self.assertNotIn('manifest',read.call_args.kwargs)

    def test_high_resolution_preview_reuses_frame_without_second_rtsp_connection(self):
        w = self.window
        w._canal_activo,w._video_epoch = CANAL_RTSP_DEFAULT,1
        w._service = Mock()
        frame = np.zeros((180,320,3),np.uint8)
        w._service.copiar_muestra.return_value = (frame,1,time.monotonic(),1)
        w._btn_capturar.setEnabled(True)
        with patch.object(w,'_procesar_captura_alta') as process,patch('ui.main_window.CameraService.capturar_frame_rtsp') as opening:
            w._capturar()
            opening.assert_not_called()
            process.assert_called_once()
            np.testing.assert_array_equal(process.call_args.args[1]['frame'],frame)
            self.assertIsNone(w._captura_thread)

    def test_video_default_never_queues_detection_and_skips_repeated_conversion(self):
        w = self.window
        w._ocr_listo,w._video_epoch = True,1
        w._service = Mock()
        w._service.copiar_muestra.side_effect = [(np.zeros((180,320,3),np.uint8),1,time.monotonic(),1),
                                                (None,1,time.monotonic(),1)]
        with patch.object(w._video,'mostrar_frame') as display,patch.object(w._ocr,'pedir_video') as infer:
            w._tick_video()
            w._tick_video()
            self.assertEqual(display.call_count,1)
            infer.assert_not_called()

    def test_both_applications_display_same_full_visual_observation(self):
        w = self.window
        w._generation,w._video_epoch = 3,2
        w._service = Mock()
        text = 'UNKNOWN ZX-591874Q EXTRA'
        result = RecognitionResult('fixture','full_photo',status='texto_reconocido',recognized_text=text,
                                   suggested_text=text,confidence=.96,selected_variant='BGR')
        m = MotorWindow()
        self.addCleanup(m.deleteLater)
        m._imagenes,m._indice_actual,m._procesando = [Path('fixture.png')],0,True
        m._inicio_imagen = time.perf_counter()
        m._tabla.setRowCount(1)
        for column in range(7):
            m._tabla.setItem(0,column,QTableWidgetItem(''))
        with tempfile.TemporaryDirectory() as tmp,patch('ui.main_window.asegurar_carpeta_capturas',return_value=Path(tmp)):
            w._on_ocr_analisis(dict(analysis=result,context=dict(generation=3,stream_epoch=2,evidence={},report=str(Path(tmp)/'ocr.json'))))
        m._on_analisis_detallado(dict(analysis=result,context=None))
        m._on_ocr_resultado(text,.96,'BGR',.96)
        self.assertEqual(w._txt_ocr.toPlainText(),m._tabla.item(0,1).text())
        self.assertEqual(w._txt_ocr.toPlainText(),text)
        self.assertIn('sin manifiesto',w._lbl_comparacion.text())

    def test_motor_export_keeps_visual_evidence_beside_excel(self):
        m = MotorWindow()
        self.addCleanup(m.deleteLater)
        m._imagenes,m._indice_actual,m._procesando = [Path('fixture.png')],0,True
        m._inicio_imagen = time.perf_counter()
        m._tabla.setRowCount(1)
        for column in range(7):
            m._tabla.setItem(0,column,QTableWidgetItem(''))
        text = 'UNKNOWN ZX-591874Q EXTRA'
        result = RecognitionResult('fixture','full_photo',status='texto_reconocido',recognized_text=text,
                                   suggested_text=text,raw_observations=[dict(text=text)],confidence=.96)
        m._on_analisis_detallado(dict(analysis=result,context=None))
        m._on_ocr_resultado(text,.96,'BGR',.96)
        with tempfile.TemporaryDirectory() as tmp,patch('motor.QFileDialog.getSaveFileName',return_value=(str(Path(tmp)/'export.xlsx'),'')),patch('motor.QMessageBox.information'):
            m._exportar_excel()
            self.assertTrue((Path(tmp)/'export.xlsx').exists())
            files = list(Path(tmp).glob('export_evidence_*.json'))
            self.assertEqual(len(files),1)
            saved = json.loads(files[0].read_text(encoding='utf8'))
            self.assertEqual(saved['records'][0]['analysis']['recognized_text'],text)
            self.assertEqual(saved['records'][0]['analysis']['raw_observations'][0]['text'],text)

    def test_stale_detection_and_old_capture_are_ignored(self):
        w = self.window
        w._generation,w._video_epoch = 3,2
        w._service = Mock()
        for gen,epoch,age in ((2,2,0),(3,1,0),(3,2,9)):
            w._on_regiones(dict(generation=gen,stream_epoch=epoch,captured_at=time.monotonic()-age,regions=[]))
            self.assertIsNone(w._region_packet)
        with patch.object(w._ocr,'pedir_ocr') as infer:
            w._procesar_captura_alta(2,dict(frame=None,stream_epoch=2))
            w._procesar_captura_alta(3,dict(frame=None,stream_epoch=1))
            infer.assert_not_called()

    def test_key_release_stops_even_after_focus_moves_to_input(self):
        w = self.window
        w._service = Mock()
        w._held_ptz.add(Qt.Key.Key_J)
        event = QKeyEvent(QEvent.Type.KeyRelease,Qt.Key.Key_J,Qt.KeyboardModifier.NoModifier)
        with patch.object(w,'_atajos_globales_activos',return_value=False),patch.object(w._ptz,'enviar') as send:
            self.assertTrue(w.eventFilter(QLineEdit(),event))
            send.assert_called_once_with(0,0,0,0,0)

    def test_audit_precedes_state_change_and_later_reading_never_downgrades(self):
        w = self.window
        w._manifiesto = [RegistroEsperado(precinto_esperado='B 0097312')]
        d = Detection('B 0097312','B 0097312',.98,[[1,1],[90,1],[90,20],[1,20]],'fixture',0,'fixture')
        c = Candidate('B 0097312',.98,.98,.98,2,True,[],[d])
        analysis = AnalysisResult('fixture','seal_roi',status='reconocido',confirmed_identifier=c.text,candidates=[c])
        w._evidence = {'fixture':True}
        with tempfile.TemporaryDirectory() as tmp,patch('ui.main_window.asegurar_carpeta_capturas',return_value=Path(tmp)):
            with patch('ui.main_window.append_decision',side_effect=OSError('fixture')):
                with self.assertRaises(OSError):
                    w._actualizar_manifiesto_por_ocr(analysis)
            self.assertEqual(w._manifiesto[0].estado,'Pendiente')
            w._actualizar_manifiesto_por_ocr(analysis)
            self.assertEqual(w._manifiesto[0].estado,'Llegado')
            analysis.status,analysis.confirmed_identifier = 'ambiguo',None
            w._actualizar_manifiesto_por_ocr(analysis)
            self.assertEqual(w._manifiesto[0].estado,'Llegado')
            events = [json.loads(s) for s in (Path(tmp)/'decisiones.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual([e['accepted'] for e in events],[True,False])

    def test_close_keeps_running_workers_alive(self):
        w = self.window
        event = Mock()
        with patch.object(w._ocr,'isRunning',return_value=True),patch('ui.main_window.QTimer.singleShot') as later:
            w.closeEvent(event)
            event.ignore.assert_called_once()
            event.accept.assert_not_called()
            later.assert_called()

    def test_queued_error_from_previous_connection_does_not_affect_new_capture(self):
        w = self.window
        w._generation,w._video_epoch = 3,2
        w._service = Mock()
        w._ocr_ocupado = True
        with patch.object(w,'_on_ocr_error') as show_error:
            for generation,epoch in ((2,2),(3,1)):
                w._on_ocr_failed_context(dict(context=dict(generation=generation,stream_epoch=epoch),error='fixture'))
            show_error.assert_not_called()
        self.assertTrue(w._ocr_ocupado)

    def test_close_cancels_deferred_startup(self):
        self.window._closing = True
        with patch.object(self.window,'_abrir_config') as config,patch.object(self.window,'_conectar') as connect:
            # Llamar la implementación real: _inicio se sustituye en setUp.
            self.patches[2].stop()
            self.window._inicio()
            config.assert_not_called()
            connect.assert_not_called()

    def test_display_confidence_belongs_to_selected_identifier(self):
        w = self.window
        w._generation,w._video_epoch = 3,2
        w._service = Mock()
        w._manifiesto = [RegistroEsperado(precinto_esperado='B 0097312')]
        wrong = Candidate('0097312',.99,.99,.99,2,True,[],[])
        selected = Candidate('B 0097312',.94,.94,.94,2,True,[],[])
        analysis = AnalysisResult('fixture','seal_roi',status='reconocido',confirmed_identifier=selected.text,
                                   suggested_text=wrong.text,candidates=[wrong,selected])
        with tempfile.TemporaryDirectory() as tmp,patch('ui.main_window.asegurar_carpeta_capturas',return_value=Path(tmp)):
            packet = dict(analysis=analysis,context=dict(generation=3,stream_epoch=2,evidence={},report=str(Path(tmp)/'ocr.json')))
            w._on_ocr_analisis(packet)
        self.assertEqual(w._txt_ocr.toPlainText(),'B 0097312')
        self.assertIn('94.0',w._lbl_confianza.text())


if __name__=='__main__':
    unittest.main()
