import datetime
import ntpath
import sys
import time
import traceback

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv

sys.path.append("/app/cv")

from utils import multy_insert, execute_sql
from utils import DBConnection


COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])

class VideoHandler:
    """
    Инициализирует PolygonHandler объект

    Параметры:
        source_weights_path     (str)   -> путь до весов модели
        source_video_path       (str)   -> путь до видео-источника
        target_video_path       (str)   -> путь сохранения файла
        confidence_threshold    (float) -> степень уверенности модели при детекции
        iou_threshold           (float) -> значение метрики Intersection over Union
        is_streaming            (bool)  -> опция остановки процесса видео
        save_db                 (bool)  -> сохранять ли данные в бд
        show_video              (bool)  -> показывать ли обработку видео в реальном времени
        class_id                (list)  -> список классов для задания разным трекерам (временно не используется)
    """

    def __init__(
            self,
            source_weights_path: str,
            source_video_path: str,
            target_video_path: str = None,
            confidence_threshold: float = 0.3,
            iou_threshold: float = 0.7,
            is_streaming: bool = True,
            save_db: bool = True,
            show_video: bool = True,
            class_id: list = [0, 1, 2],
    ) -> None:
        self.model = YOLO(source_weights_path)
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.is_streaming = is_streaming
        self.save_db = save_db
        self.frame = None
        self.show_video = show_video
        self.class_id = class_id

        self.data_trace = []  # Данные о трассировках

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)

        self.tracker = sv.ByteTrack()

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )

        id_query = execute_sql(
            f'select max(id) from {DBConnection.SQL_SCHEME}.{DBConnection.RECOGNITION_DATA}',
            params={},
            is_select=True
        )

        self.id = 0 if id_query[0]['max'] is None else id_query[0]['max'] + 1

    def send_frame(self, frame):
        """
        Передает файлы в self.frame для дальнейшей передачи в Response
        """
        _, jpeg = cv2.imencode('.jpg', frame)
        self.frame = jpeg.tobytes()

    def get_frame(self):
        """
        Генератор поочередного отображения фреймов в stream
        """
        while True:
            if self.frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + self.frame + b'\r\n')
            time.sleep(0.01)

    def stop_stream(self):
        """
        Остановка потока stream и обработки видео
        """
        self.is_streaming = False  # Остановка потока

    def process_video(self):
        """
        Функция обработки видео с учетом:
        self.show_video         -> наблюдать процесс обработки видео / не наблюдать
        self.target_video_path  -> сохранять конечное обработанное видео / не сохранять
        self.is_streaming       -> опция остановки процесса видео
        """

        # Инициализация генератора фреймов
        generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        try:
            if self.target_video_path:
                with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                    for i, frame in enumerate(tqdm(generator, total=self.video_info.total_frames), start=1):
                        annotated_frame = self.process_batch([frame], [i])[0]
                        # Отправляем фрейм в start_stream
                        if self.show_video:
                            self.send_frame(annotated_frame)
                        sink.write_frame(annotated_frame)
                        # Если кнопка stop_stream не была нажата, идем дальше    
                        if self.is_streaming:
                            continue
                        else:
                            break
            else:
                for i, frame in enumerate(tqdm(generator, total=self.video_info.total_frames), start=1):
                    annotated_frame = self.process_batch([frame], [i])[0]
                    # Отправляем фрейм в start_stream
                    if self.show_video:
                        self.send_frame(annotated_frame)
                    # Если кнопка stop_stream не была нажата, идем дальше    
                    if self.is_streaming:
                        continue
                    else:
                        break

        except Exception as e:
            traceback.print_exc()

        finally:
            if self.save_db:

                # Запись объектов детекции
                multy_insert(self.data_trace,
                             ["id", "x_center", "y_center", "class_id", "class_name", "confidence", "tracker_id",
                              "current_frame", "total_frames", "file_name", "time"],
                             DBConnection.SQL_SCHEME,
                             DBConnection.RECOGNITION_DATA)

    def process_batch(self, frames: list, frame_numbers: list) -> list:
        # Прохождение одного фрейма через модель и вычленение детекций
        results = self.model(frames, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold,
                             imgsz=[736, 1280])

        annotated_frames = []
        for frame, result, frame_number in zip(frames, results, frame_numbers):
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[(np.isin(detections.class_id, self.class_id))]
            detections = self.tracker.update_with_detections(detections)

            for i in range(len(detections.xyxy)):
                # Массив данных для выгрузки в базу
                self.data_trace.append((
                    int(self.id),
                    (float(detections.xyxy[i][0]) + float(detections.xyxy[i][2])) / 2,
                    (float(detections.xyxy[i][1]) + float(detections.xyxy[i][3])) / 2,
                    int(detections.class_id[i]),
                    str(detections.data['class_name'][i]),
                    float(detections.confidence[i]),
                    int(detections.tracker_id[i]),
                    int(frame_number),
                    int(self.video_info.total_frames),
                    ntpath.basename(self.source_video_path),
                    datetime.datetime.now(),
                ))

            frame = self.annotate_frame(frame, detections)
            annotated_frames.append(frame)
        return annotated_frames

    def annotate_frame(
            self,
            frame: np.ndarray,
            detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()

        # Задание надписи для каждого объекта
        labels = [f"{tracker_id} {dtype['class_name']} {confidence:0.2f}"
                  for xyxy, mask, confidence, class_id, tracker_id, dtype
                  in detections]
        
        # Отрисовка аннотаций
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)

        return annotated_frame