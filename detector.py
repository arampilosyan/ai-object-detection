from pathlib import Path
from ultralytics import YOLO


class ObjectDetector:
    """Thin wrapper around Ultralytics YOLO object detection."""

    def __init__(self, model_name: str = "yolo11n.pt", confidence: float = 0.25):
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect_image(self, image_path: str | Path, output_path: str | Path):
        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence,
            verbose=False,
        )
        results[0].save(filename=str(output_path))
        return self._get_detections(results[0])

    def detect_video(self, video_path: str | Path, output_path: str | Path):
        """Process a video and save annotated output to the requested path."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise ValueError("Could not create output video.")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = self.model.predict(
                    source=frame,
                    conf=self.confidence,
                    verbose=False,
                )[0]
                writer.write(result.plot())
        finally:
            cap.release()
            writer.release()

        return output_path

    def detect_frame(self, frame):
        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
        )[0]
        return result.plot(), self._get_detections(result)

    @staticmethod
    def _get_detections(result):
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            detections.append({
                "label": result.names[class_id],
                "confidence": round(float(box.conf[0].item()) * 100, 2),
                "x1": round(float(box.xyxy[0][0].item()), 1),
                "y1": round(float(box.xyxy[0][1].item()), 1),
                "x2": round(float(box.xyxy[0][2].item()), 1),
                "y2": round(float(box.xyxy[0][3].item()), 1),
            })
        return detections


# OpenCV is imported here to keep the video implementation explicit.
import cv2
