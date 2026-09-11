# Eye Blink Rate Detection (Fatigue Monitoring Demo)

Mini project for **image processing / computer vision**: estimate **blink rate** from a webcam or video using **dlib’s 68 facial landmarks** and the classical **Eye Aspect Ratio (EAR)** (Soukupová–Čech–style), suitable for prototyping **fatigue / attention** dashboards (research / education only — not a medical device).

## Method

1. **Face detection**: dlib frontal face detector (HOG + SVM).
2. **Landmarks**: `shape_predictor_68_face_landmarks.dat` → 68 (x, y) points.
3. **EAR** per eye: vertical distances between eyelid points vs horizontal eye width; average left + right for stability.
4. **Blink**: EAR drops below a threshold for several consecutive frames (eye closed), then rises again → one blink.
5. **Blink rate**: blinks in a rolling time window → **blinks per minute**.

## Setup

**Python 3.9+** recommended.

```bash
cd lohith_ipcv
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_model.py
```

On Windows, if `pip install dlib` fails, install [CMake](https://cmake.org/download/) and a C++ build tool (e.g. **Visual Studio Build Tools** with “Desktop development with C++”), then retry `pip install dlib`. Alternatively use **conda**: `conda install -c conda-forge dlib`.

## Run

```bash
python main.py
```

- **Quit**: press `q` in the video window.
- **Webcam index**: `python main.py --camera 1`
- **Video file**: `python main.py --video path\to\clip.mp4`
- **Tuning**: `python main.py --ear-thresh 0.22 --ear-consec 3`

## Project layout

| File | Role |
|------|------|
| `main.py` | OpenCV loop, blink counting, rolling BPM |
| `ear_utils.py` | EAR math and landmark eye indices |
| `download_model.py` | Fetches dlib’s 68-point model into `models/` |
| `models/` | Contains `.dat` after download (ignored by git) |

## GitHub

```bash
git init
git add .
git commit -m "Add EAR blink detection with dlib 68 landmarks"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

Do **not** commit `models/*.dat` (large binary); collaborators run `python download_model.py`.

## References

- T. Soukupová and J. Čech, *Real-Time Eye Blink Detection Using Facial Landmarks*, 2016.
- [dlib.net](http://dlib.net/) — shape predictor file.

## License

Use and modify freely for coursework / demos; attribute dlib and cited papers where appropriate.
