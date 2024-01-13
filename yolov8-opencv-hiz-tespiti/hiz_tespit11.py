import supervision as sv
import cv2
from ultralytics import YOLO

if __name__ == "__main__":

    video_info = sv.VideoInfo.from_video_path(video_path="cctv_trafik.mp4")
    model = YOLO("yolov8n.pt")

    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=4)

    frame_generator = sv.get_video_frames_generator(source_path="cctv_trafik.mp4")

    for frame in frame_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )

        outfile = 'resim_1.jpg'
        cv2.imwrite(outfile, annotated_frame)

        cv2.imshow("frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()