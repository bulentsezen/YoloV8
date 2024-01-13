# farklı video görüntüsü ile hız tespiti
# video_path="cctv_trafik.mp4"
# yeni videodan resim_1.jpg üret (hiz_tespiti11.py) ve koordinatları belirle
# SOURCE = np.array([[310, 190], [630, 190],[740, 720], [-70, 720]])
# TARGET_WIDTH = 20
# TARGET_HEIGHT = 30

import supervision as sv
import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict, deque

SOURCE = np.array([[310, 190], [630, 190],[740, 720], [-70, 720]])

TARGET_WIDTH = 20
TARGET_HEIGHT = 30

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        print("points :", points)
        if len(points) != 0:

            reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
            transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)

            return transformed_points.reshape(-1, 2)

        else:
            hata_giderici_gecici_deger = np.array([[1, 2], [5, 8]], dtype='float')
            return hata_giderici_gecici_deger


if __name__ == "__main__":

    video_info = sv.VideoInfo.from_video_path(video_path="cctv_trafik.mp4")
    model = YOLO("yolov8n.pt")

    byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_thresh=0.5)

    thickness = 3
    text_scale = 1

    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER, color_lookup=sv.ColorLookup.TRACK)

    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER, color_lookup=sv.ColorLookup.TRACK)

    frame_generator = sv.get_video_frames_generator(source_path="cctv_trafik.mp4")

    polygon_zone = sv.PolygonZone(polygon=SOURCE, frame_resolution_wh=video_info.resolution_wh)

    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    for frame in frame_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)

        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)

        labels = []
        for tracker_id in detections.tracker_id:
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6
                labels.append(f"#{tracker_id} {int(speed)} km/h")

        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.red())
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections)

        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels)

        cv2.imshow("frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()