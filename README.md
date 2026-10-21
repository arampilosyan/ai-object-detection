# Vision AI — Object Detection

A Python computer-vision application that detects objects in images, processes videos, and supports a live browser camera using YOLO and Streamlit.

## Features

- Image detection with annotated output and downloadable results
- Video processing with downloadable annotated video
- Live camera object detection in supported browsers
- Adjustable confidence threshold
- Detection table with class, confidence, and bounding-box coordinates
- Small, modular codebase suitable for a learning portfolio

## Tech stack

- Python
- Ultralytics YOLO (`yolo11n.pt`)
- OpenCV
- Streamlit
- streamlit-webrtc

## Setup

Python 3.10–3.12 is recommended.

```bash
git clone https://github.com/YOUR_USERNAME/ai-object-detection.git
cd ai-object-detection

python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the model weights. An internet connection is needed for this initial download.

## How to use

1. Open the local Streamlit URL shown in the terminal.
2. Choose the Image, Video, or Live camera tab.
3. Upload a file or allow camera access.
4. Review detections and download the annotated output.

## Notes

- Live camera requires browser camera permission. Hosted use generally requires HTTPS and compatible WebRTC networking.
- Video encoding depends on the OpenCV build and available codecs.
- Detection confidence is not a guarantee of correctness.
- This repository is a demo/learning project, not a safety-critical monitoring system.
- Check the model and dataset licenses and the applicable usage terms before commercial use.

## Project structure

```text
ai-object-detection/
├── app.py
├── detector.py
├── requirements.txt
├── README.md
├── .gitignore
└── tests/
    └── test_detector.py
```

## License

Choose and add a license before accepting contributions or distributing the project.
