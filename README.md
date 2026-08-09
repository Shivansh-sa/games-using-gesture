# 🎮 Games Using Gesture — Hands-Free Game Control

Play popular games **without touching a keyboard** — using your **body movements** and **hand gestures** captured through a webcam. This project uses real-time **pose estimation** and **hand tracking** to translate physical movement into in-game key presses.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=flat&logo=google&logoColor=white)
![cvzone](https://img.shields.io/badge/cvzone-00A98F?style=flat)

---

## 🕹️ Games Included

### 1️⃣ Subway Surfers / Temple Run — Full-Body Pose Control
Uses **MediaPipe Pose** to track your shoulders and body position, then maps your movement to the endless-runner controls.

| Your Movement | Detected As | Key Pressed |
|---------------|-------------|-------------|
| Lean left | `Left` | ← Left arrow |
| Lean right | `Right` | → Right arrow |
| Jump up | `Jumping` | ↑ Up arrow |
| Crouch down | `Crouching` | ↓ Down arrow |
| Stand normally | `Standing` | — |

The game auto-calibrates your standing position when you first step into frame.

### 2️⃣ Hill Climb Racing — Hand Gesture Control
Uses **cvzone's HandDetector** to read your fingers and control the car.

| Your Gesture | Key Pressed |
|--------------|-------------|
| ✊ Closed fist | ← Left (brake / reverse) |
| ✋ Open palm | → Right (accelerate) |
| No hand detected | Keys released |

---

## 🛠️ Tech Stack

**Language:** Python
**Computer Vision:** OpenCV, MediaPipe, cvzone
**Input Automation:** pyautogui, pynput

---

## ⚙️ How It Works

1. **Capture** — OpenCV reads the live webcam feed frame by frame.
2. **Detect** — MediaPipe (pose) or cvzone (hand) identifies landmarks in each frame.
3. **Interpret** — Simple geometric rules classify the movement (e.g. shoulder midpoint above a threshold = *Jumping*; all fingers down = *fist*).
4. **Act** — `pyautogui` / `pynput` sends the matching keyboard key to the active game window.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Shivansh-sa/games-using-gesture.git
cd games-using-gesture

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```
opencv-python
mediapipe
cvzone
pyautogui
pynput
```

---

## 🚀 Usage

1. Open the game you want to play (Subway Surfers / Temple Run in a browser or emulator, or Hill Climb Racing).
2. Launch the notebook:
   ```bash
   jupyter notebook "GAME control using GESTURE.ipynb"
   ```
3. Run the cell for the game you want:
   - **Subway Surfers / Temple Run** → run the *TEMPLE RUN* cell
   - **Hill Climb Racing** → run the *HILL CLIMB RACING* cell
4. **Keep the game window focused** so it receives the key presses.
5. Stand back so your upper body (or hand) is visible to the camera and start moving!

> Press **ESC** (Subway Surfers) or **Q** (Hill Climb) to stop.

---

## 💡 Tips for Best Results

- Use a **well-lit room** and a plain background for cleaner detection.
- For pose control, make sure your **shoulders and upper body** are in frame.
- Keep the **game window in focus** — key presses go to whatever window is active.
- Sit/stand at a comfortable distance so movements are clearly visible.

---

## 🔮 Future Improvements

- Add more games and gesture mappings
- On-screen calibration and sensitivity controls
- Support for more gestures (e.g. tilt, two-hand controls)
- Package as a single launcher with a game-selection menu

---

## 👤 Author

**Shivansh Srivastava**
🔗 GitHub: [github.com/Shivansh-sa](https://github.com/Shivansh-sa)
💼 LinkedIn: [Shivansh Srivastava](https://www.linkedin.com/in/shivansh-srivastava-3b593928b/)
📧 srivastavashivansh8922@gmail.com

---

⭐ *If you enjoyed this project, consider giving it a star on GitHub!*
