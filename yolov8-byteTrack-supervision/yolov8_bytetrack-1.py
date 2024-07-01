# supervision box_annotator ile nesne tespiti

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
        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.box_annotator = sv.BoxAnnotator(color=COLORS)

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path)

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
        return self.annotate_frame(frame, detections)

    def annotate_frame(
            self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        return annotated_frame


if __name__ == "__main__":
    processor = VideoProcessor(
        source_video_path="cctv_trafik.mp4",
    )
    processor.process_video()












