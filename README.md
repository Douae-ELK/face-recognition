# Face Recognition Attendance System

A real-time face recognition system built with Python, OpenCV, and the `face_recognition` library. It identifies people through a webcam, categorizes them (Student, Professor, Administrative Staff), and records attendance in a SQLite database.

## Features

- Live face detection and recognition from webcam
- Train new faces directly from the camera (10 samples per person)
- Color-coded boxes by category
- Attendance tracking (first seen, last seen, total appearances)
- Alerts and photo capture for unknown faces
- Statistics view and daily CSV export

## Requirements

- Python 3.8+
- A webcam
- Libraries: `opencv-python`, `numpy`, `face_recognition` (requires `dlib`)

## Installation

```bash
pip install -r requirements.txt
```

> On Windows, installing `dlib` may require CMake and Visual Studio Build Tools.

## Usage

```bash
python face_recognition_advanced.py
```

### Keyboard controls

| Key | Action |
|-----|--------|
| `N` | Add a new face |
| `T` | Show statistics |
| `E` | Export today's attendance to CSV |
| `Q` | Quit |

**Training a new face:** press `N`, type the name in the terminal, choose a category (`S` Student, `P` Professor, `A` Administrative Staff), then press `SPACE` to capture samples and `ENTER` to finish.

### Box colors

- Green: Student
- Gold: Professor
- Blue: Administrative Staff
- Red: Unknown

## Project structure

```
face_recognition_advanced.py   # Main application
requirements.txt               # Python dependencies
face_data/                     # Auto-created: database, encodings, photos
attendance_YYYY-MM-DD.csv      # Exported reports
```

## Privacy

This project stores facial data. Only use it with the consent of the people being recorded and respect local data protection laws.
