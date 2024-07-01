# trace-annotator aktifleştirildi

from ultralytics import YOLO
import numpy as np
import cv2
import supervision as sv
from tqdm import tqdm

COLORS = sv.ColorPalette.default()

class VideoProcessor:
    def __init__(
        self,
        source_video_path: str,
    ) -> None:
        self.conf_threshold = 0.5
        self.iou_threshold = 0.5
        self.source_video_path = "cctv_trafik.mp4"

        self.model = YOLO('yolov8n.pt')
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)

        self.bounding_box_annotator = sv.BoundingBoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.blue()
        )

        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        for frame in tqdm(frame_generator, total=self.video_info.total_frames):
            annotated_frame = self.process_frame(frame)
            cv2.imshow("Processed Video", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        detections.class_id = np.zeros(len(detections))
        detections = self.tracker.update_with_detections(detections)

        return self.annotate_frame(frame, detections)

    def annotate_frame(
            self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)

        annotated_frame = self.bounding_box_annotator.annotate(
            annotated_frame, detections
        )
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections, labels
        )
        return annotated_frame


if __name__ == "__main__":
    processor = VideoProcessor(
        source_video_path="cctv_trafik.mp4",
    )
    processor.process_video()












