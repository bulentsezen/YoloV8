## perspektif görünümü opencv getPerspectiveTransform ile doğrultulmuş düz görünümlü matrise dönüştürme
## bu dönüştürülmüş matrisi kullanarak araçların yaklaşık x,y koordinatlarını hesaplama

## view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
# points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
# points = view_transformer.transform_points(points=points).astype(int)

# labels = [f"x: {x}, y: {y}"
#             for [x,y] in points]


import supervision as sv
import cv2
from ultralytics import YOLO
import numpy as np

SOURCE = np.array([[int(1252*0.3), int(787*0.3)], [int(2298*0.3), int(803*0.3)],
                   [int(5039*0.3), int(2159*0.3)], [int(-550*0.3), int(2159*0.3)]])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

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
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


if __name__ == "__main__":

    video_info = sv.VideoInfo.from_video_path(video_path="vehicles.mp4")
    model = YOLO("yolov8n.pt")

    byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_thresh=0.5)

    thickness = 3
    text_scale = 1

    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER)

    frame_generator = sv.get_video_frames_generator(source_path="vehicles.mp4")

    polygon_zone = sv.PolygonZone(polygon=SOURCE, frame_resolution_wh=video_info.resolution_wh)

    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    for frame in frame_generator:
        half_frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
        result = model(half_frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)

        labels = [
            f"x: {x}, y: {y}"
            for [x,y] in points]

        # for tracker_id in np.array(detections.tracker_id):
        #     labels.append(f"#{tracker_id}")

        annotated_frame = half_frame.copy()
        annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.red())
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections)

        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels)

        cv2.imshow("frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()