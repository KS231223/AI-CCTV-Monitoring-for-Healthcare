# AI CCTV Monitoring Backend (FastAPI + Computer Vision)

This project implements a **real-time AI-powered CCTV backend system** for **person detection, identification, tracking, and biometric analysis** using computer vision and machine learning.

---

## 🚀 Features

* Real-time CCTV video processing
* Human detection using YOLO
* Face recognition with embeddings (FaceNet + Pinecone)
* Pose tracking and gait analysis (MediaPipe)
* Remote heart rate estimation (rPPG)
* Multi-threaded processing pipeline
* WebSocket video streaming
* Firebase integration for user data

---

## 🧠 System Overview

The system processes live video frames and performs:

1. **Detection** → Identify people in frame
2. **Tracking** → Match detections to existing persons
3. **Identification** → Recognize faces using embeddings
4. **Analysis** → Extract:

   * Gait features
   * Movement velocity
   * Heart rate (rPPG)

---

## 🧩 Architecture

### 🔹 Core Components

#### 1. CCTV Service (`cctv_service.py`)

* Main processing loop
* Runs detection, tracking, and analysis
* Streams frames via WebSocket

#### 2. Frame Processor

* Uses MediaPipe Pose Landmarker
* Extracts:

  * Joint positions
  * Knee angles
  * Torso metrics
  * Movement velocity

#### 3. Person Model

Tracks:

* Bounding box
* Identity (if recognized)
* Frame history (last 120 frames)
* Gait + biometric data

---

## 🔄 Processing Pipeline

### Step 1: Detection

* YOLO model detects humans in each frame

### Step 2: Matching

* Bounding boxes matched to existing tracked persons
* Distance-based tracking (`MAX_DIST` threshold)

### Step 3: Identification

* Face extracted using MTCNN
* Embedding generated using FaceNet
* Compared against Pinecone vector database

---

### Step 4: Parallel Analysis (Threads)

Each identified person spawns:

#### 🧍 Gait Thread

* Uses pose landmarks
* Tracks movement patterns
* Updates frame buffer

#### ❤️ rPPG Thread

* Extracts forehead region
* Estimates heart rate from color variations
* Requires ~60 frames for stable output

---

## 📡 API Endpoints

### WebSocket Stream

```id="ws1"
GET /cctv/{cctv_id}/ws
```

* Streams live processed frames (JPEG)
* Includes overlays (bounding boxes, labels, HR)

---

### Get Tracked Persons

```id="api1"
GET /cctv/{cctv_id}/persons
```

Returns:

* List of tracked persons
* Includes identity + metrics

---

## 🔌 Technologies Used

* **FastAPI** → backend framework
* **OpenCV** → video processing
* **Ultralytics YOLO** → object detection
* **FaceNet (facenet-pytorch)** → face embeddings
* **MTCNN** → face detection
* **MediaPipe** → pose estimation
* **Firebase Firestore** → user database
* **Pinecone** → vector similarity search
* **NumPy / PyTorch** → ML computations

---

## ⚙️ Setup & Installation

### 1. Install Dependencies

```bash id="dep1"
pip install fastapi uvicorn opencv-python numpy torch torchvision
pip install ultralytics facenet-pytorch mediapipe pinecone-client firebase-admin python-dotenv
```

---

### 2. Environment Variables

Create a `.env` file:

```env id="env1"
PINECONE_API_KEY=your_api_key_here
```

---

### 3. Firebase Setup

* Add `serviceAccountKey.json` to project root
* Ensure Firestore is enabled

---

### 4. Run Server

```bash id="run1"
uvicorn main:app --reload
```

---

## 🎥 How It Works (Simplified)

1. Webcam feed is captured (`cv2.VideoCapture`)
2. YOLO detects people
3. Each person:

   * Matched or newly created
   * Sent to ID thread (face recognition)
4. Identified persons:

   * Spawn gait + rPPG threads
5. Results:

   * Stored in memory (`tracked_persons`)
   * Streamed to frontend

---

## 📊 Data Tracked Per Person

* `person_id`
* Bounding box
* Velocity
* Gait features:

  * Knee angles
  * Torso position
  * Shoulder sway
* Heart rate (if available)

---

## ⚠️ Notes

* Designed for **real-time performance**, but hardware dependent
* Threading is used for:

  * Identification
  * Gait analysis
  * rPPG processing
* Face recognition requires pre-stored embeddings in Pinecone
* rPPG accuracy depends on:

  * Lighting conditions
  * Camera quality
  * Frame stability

---

## 🧩 Limitations

* No persistent tracking across restarts
* Basic distance-based tracking (no DeepSORT yet)
* Single-camera support
* No authentication for API endpoints

---

## 🔮 Future Improvements

* Multi-camera support
* Better tracking (DeepSORT / ByteTrack)
* Database persistence
* Frontend dashboard
* Alerts / anomaly detection
* Improved biometric accuracy

---

## 📜 License

For educational and experimental use. Add a proper license if deploying publicly.

---

## 👨‍💻 Author Notes

* Designed as a **modular AI pipeline**
* Focus on combining:

  * Detection
  * Identification
  * Biometric analysis
* Threads are used to decouple heavy processing tasks

---

## ⚠️ Notes & Current Limitations

### 🎯 Accuracy Overview

* **Face Recognition (High Accuracy ✅)**

  * The facial recognition pipeline performs **reliably and consistently**
  * Embeddings generated using FaceNet and stored in Pinecone provide:

    * Strong identity matching
    * Good tolerance to minor variations (angle, lighting)
  * Overall, this is currently the **most accurate and stable component** of the system

---

### ❤️ rPPG (Heart Rate Estimation) – Experimental ⚠️

* The rPPG module is **not fully reliable yet**
* Accuracy is affected by:

  * Lighting conditions
  * Camera quality
  * Subject movement
  * ROI (forehead) stability

#### Current Implementation

* Uses a **fusion of two classical methods**:

  * Chrominance-based method
  * Green channel method
* Frequency analysis (FFT) is applied to estimate heart rate

#### Important Limitation

* Results can be **noisy and inconsistent**
* Variance-based weighting helps, but is not sufficient for stability

---

### 🧪 Temporary Fix (Clamping)

To keep outputs within a realistic range, a **manual clamp is currently applied**:

```python
hr_fused = max(60, min(100, hr_fused))
```

* This ensures heart rate stays within **human resting range (60–100 BPM)**
* A small random jitter is also added:

```python
hr_fused += random.uniform(-5, 5)
```

#### ⚠️ Implication

* The displayed heart rate is **not medically accurate**
* It should be treated as:

  > *a rough, demo-level estimate rather than a precise measurement*

---

### 📉 Overall System Accuracy

| Component         | Status      | Notes                   |
| ----------------- | ----------- | ----------------------- |
| Face Recognition  | ✅ High      | Most reliable part      |
| Person Detection  | ✅ Good      | YOLO performs well      |
| Tracking          | ⚠️ Moderate | Simple distance-based   |
| Gait Analysis     | ⚠️ Moderate | Depends on pose quality |
| rPPG (Heart Rate) | ❌ Low       | Experimental + clamped  |

---

### 🧠 Key Takeaway

* The system is **strong in identification (who the person is)**
* But still **developing in physiological analysis (what their body signals are)**

---

### 🔮 Planned Improvements for rPPG

* Better ROI extraction (face/forehead stabilization)
* Temporal filtering (bandpass filters)
* Signal detrending
* Use of deep learning-based rPPG models
* Removal of artificial clamping once stable

---

## 🧾 Summary

> This system is currently **best suited for identity tracking and recognition**,
> while biometric signals like heart rate are still **experimental and under active development**.

---

