# Third Eye 👁️

**An experimental AI-powered visual assistance prototype for blind and visually impaired people.**

Third Eye combines real-time object detection, object tracking, proximity-based alerts, and AI-powered scene understanding to help users interpret their surroundings.

## Features

- **Object detection:** Identifies objects using YOLO11.
- **Object tracking:** Tracks detected objects across video frames using ByteTrack.
- **Proximity-based alerts:** Uses bounding-box size, position, and changes over time to identify potentially concerning obstacles.
- **Scene understanding:** Integrates Gemma 3 through Ollama to describe camera images.
- **Voice interaction:** Includes a speech-recognition and spoken-response pipeline that is still being debugged.
- **Camera input:** Accepts a phone camera stream while processing video on a computer.
- **Optional cloud mode:** Supports a Gemini API backend.

## Tech Stack

Python · OpenCV · Ultralytics YOLO · ByteTrack · Ollama · Gemma 3 · Faster-Whisper

## Requirements

- Windows computer
- Python environment compatible with the required dependencies
- A phone camera streaming video over a reachable network or USB/ADB forwarding
- Ollama installed separately, with the configured vision model available
- Optional Gemini API key for cloud mode

## Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the Python dependencies.
4. Install and start Ollama.
5. Download the model configured in `main.py`.
6. Configure `SOURCE` in `main.py` to match your phone camera stream.
7. Run the application.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Download the default local model:

```bash
ollama pull gemma3:4b
```

Run the application:

```bash
python main.py
```

The default camera address in the code is specific to the original development setup and must be changed to match your own device.

## Controls

| Key | Action |
|---|---|
| `d` | Request a scene description |
| `a` | Record a voice question |
| `c` | Switch between local and cloud backends |
| `s` | Toggle automatic scene descriptions |
| `q` | Quit the application |

Click the video window before using keyboard shortcuts.

## Current Development Status

This is a laptop-based experimental prototype. The spoken feedback pipeline requires debugging, and the Google Maps destination-selection and navigation features are planned for future development rather than implemented features.

## Future Improvements

- Stabilize speech recognition and spoken responses.
- Improve detection and alert reliability.
- Explore Google Maps integration for destination selection.
- Investigate mobile deployment.
- Conduct usability and safety testing with feedback from visually impaired people.

## Safety Notice

Third Eye is not a validated mobility aid. Its visual heuristics do not guarantee accurate physical distances or collision predictions. Do not rely on this prototype as the sole means of obstacle avoidance or navigation.
