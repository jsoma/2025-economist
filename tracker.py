import os
import cv2
import supervision as sv
from inference import get_model

model = get_model("yolov10n-640")

video_path = "istockphoto-534232220-640_adpp_is.mp4"
frame_generator = sv.get_video_frames_generator(video_path)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
line_zone_annotator = sv.LineZoneAnnotator(text_thickness=1)

byte_track = sv.ByteTrack()
byte_track.reset()

start = sv.Point(200, 175)
end = sv.Point(700, 175)
line_zone = sv.LineZone(start, end)
trace_annotator = sv.TraceAnnotator()
smoother = sv.DetectionsSmoother()

# video_info = sv.VideoInfo.from_video_path(video_path=video_path)
# with sv.VideoSink(target_path="output.mp4", video_info=video_info) as sink:

for frame in frame_generator:
    result = model.infer(frame, confidence=0.3)[0]
    detections = sv.Detections.from_inference(result)
    detections = byte_track.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)

    line_zone.trigger(detections)

    annotated_frame = frame.copy()

    labels = [
        f"#{tracker_id} {model.class_names[class_id]} {confidence:0.2f}"
    for _, _, confidence, class_id, tracker_id, _
    in detections
]

    annotated_frame = box_annotator.annotate(annotated_frame, detections)
    annotated_frame = trace_annotator.annotate(annotated_frame, detections)

    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels)

    annotated_frame = line_zone_annotator.annotate(
        annotated_frame,
        line_counter=line_zone)

    cv2.imshow("", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    # sink.write_frame(frame=annotated_frame)