import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

from detector import ObjectDetector

st.set_page_config(
    page_title="Vision AI | Object Detection",
    page_icon="👁️",
    layout="wide",
)

st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 2rem;}
.hero {padding: 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#101827,#1d3557); color: white;}
.hero h1 {margin: 0; font-size: 2.4rem;}
.hero p {color: #d5e2f4; margin-bottom: 0;}
</style>
<div class="hero">
  <h1>👁️ Vision AI</h1>
  <p>Object detection for images, videos, and live camera streams.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    return ObjectDetector()

with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    st.caption("Higher confidence reduces weak detections.")
    st.info("The first run downloads the YOLO model and may take a little longer.")

try:
    detector = load_detector()
    detector.confidence = confidence
except Exception as exc:
    st.error(f"Could not load the AI model: {exc}")
    st.stop()

image_tab, video_tab, camera_tab = st.tabs(
    ["🖼️ Image detection", "🎬 Video detection", "📷 Live camera"]
)

with image_tab:
    uploaded_image = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png", "webp"], key="image"
    )
    if uploaded_image:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            with tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / "input.jpg"
                output = Path(tmp) / "detected.jpg"
                image.save(source)
                detections = detector.detect_image(source, output)

                left, right = st.columns(2)
                with left:
                    st.subheader("Original image")
                    st.image(image, use_container_width=True)
                with right:
                    st.subheader("Detected objects")
                    st.image(str(output), use_container_width=True)

                st.metric("Objects detected", len(detections))
                if detections:
                    st.dataframe(detections, use_container_width=True, hide_index=True)
                else:
                    st.info("No objects detected at the selected confidence threshold.")

                st.download_button(
                    "Download annotated image",
                    data=output.read_bytes(),
                    file_name="vision-ai-detected.jpg",
                    mime="image/jpeg",
                )
        except Exception as exc:
            st.error(f"Image processing failed: {exc}")

with video_tab:
    uploaded_video = st.file_uploader(
        "Choose a video", type=["mp4", "mov", "avi", "mkv"], key="video"
    )
    if uploaded_video and st.button("Analyze video", type="primary"):
        with st.spinner("Processing video… This can take a while for long clips."):
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    source = Path(tmp) / f"input{Path(uploaded_video.name).suffix or '.mp4'}"
                    output = Path(tmp) / "vision-ai-processed.mp4"
                    source.write_bytes(uploaded_video.getvalue())
                    detector.detect_video(source, output)
                    video_bytes = output.read_bytes()
                    st.video(video_bytes)
                    st.download_button(
                        "Download processed video",
                        data=video_bytes,
                        file_name="vision-ai-processed.mp4",
                        mime="video/mp4",
                    )
            except Exception as exc:
                st.error(f"Video processing failed: {exc}")

with camera_tab:
    st.write("Allow camera access in your browser when prompted.")
    class LiveDetector(VideoTransformerBase):
        def transform(self, frame):
            import cv2
            image = frame.to_ndarray(format="bgr24")
            annotated, _ = detector.detect_frame(image)
            return annotated

    webrtc_streamer(
        key="vision-ai-live-camera",
        video_transformer_factory=LiveDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.caption("Live camera works in supported browsers. On hosted deployments, HTTPS is generally required.")
