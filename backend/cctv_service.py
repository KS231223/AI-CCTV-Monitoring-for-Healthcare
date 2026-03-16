# backend/cctv_service.py
import asyncio
import cv2
import threading
import queue
import numpy as np
from ultralytics import YOLO
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from entities import Person
from frame_processor import DecoupledProcessor, PersonAdapter
from rPPGmicroservice import rppg_simple
import firebase_admin
from firebase_admin import credentials, firestore
from pinecone import Pinecone, ServerlessSpec
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import time
import os
from dotenv import load_dotenv


# ----------------------------
# Firebase setup
# ----------------------------
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

load_dotenv()
# ----------------------------
# Pinecone setup
# ----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "faceembeddings"

if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=512,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)


# ----------------------------
# FastAPI
# ----------------------------
router = APIRouter()

# ----------------------------
# Models
# ----------------------------
yolo_model = YOLO("yolo26n.pt")
mtcnn = MTCNN(keep_all=True)
face_model = InceptionResnetV1(pretrained="vggface2").eval()
processor = DecoupledProcessor(model_path="pose_landmarker_heavy.task", fps=30)
adapter = PersonAdapter(processor)


# ----------------------------
# Globals
# ----------------------------
tracked_persons: dict[str, Person] = {}
MAX_DIST = 120
raw_frame = None
overlay_frame = None


# ----------------------------
# Embedding
# ----------------------------
def compute_embedding(face_crop):
    face = Image.fromarray(face_crop).resize((160,160))
    tensor = torch.tensor(
        np.array(face).transpose(2,0,1)
    ).float()/255.0
    tensor = tensor.unsqueeze(0)
    emb = face_model(tensor).detach().numpy()[0]
    emb /= np.linalg.norm(emb) + 1e-8
    return emb


# ----------------------------
# rPPG daemon
# ----------------------------
def run_rppg_daemon(person):
    while True:
        try:
            frame, bbox = person.rppg_queue.get(timeout=5)
        except queue.Empty:
            continue
        x1,y1,x2,y2 = bbox
        person_crop = frame[y1:y2,x1:x2]
        boxes,_ = mtcnn.detect(person_crop)
        if boxes is None:
            return
        fx1,fy1,fx2,fy2 = map(int,boxes[0])
        # approximate forehead region
        roi = frame[fy1:int(fy1+(fy2-fy1)*0.3), fx1:fx2]
        if roi.size == 0:
            continue
        # store in person buffer
        person.last_120_rPPG.append(roi)
        if len(person.last_120_rPPG) > 120:
            person.last_120_rPPG.pop(0)
        # compute HR once enough frames collected
        if len(person.last_120_rPPG) >= 60:
            hr, var = rppg_simple(person.last_120_rPPG)
            person.heart_rate = float(hr)
            person.rppg_variance = float(var)
        person.rppg_queue.task_done()


# ----------------------------
# enqueue rPPG
# ----------------------------
def _enqueue_rppg_frame(person, frame, bbox):
    if hasattr(person, "rppg_queue"):
        try:
            person.rppg_queue.put_nowait((frame.copy(), bbox))
        except queue.Full:
            pass


# ----------------------------
# ID daemon
# ----------------------------
def run_id_daemon(temp_obj):
    frame = temp_obj["frame"]
    bbox = temp_obj["bbox"]
    x1,y1,x2,y2 = bbox
    person_crop = frame[y1:y2,x1:x2]
    boxes,_ = mtcnn.detect(person_crop)
    if boxes is None:
        return
    fx1,fy1,fx2,fy2 = map(int,boxes[0])
    face_crop = person_crop[fy1:fy2,fx1:fx2]
    if face_crop.size == 0:
        return
    embedding = compute_embedding(face_crop)
    results = index.query(
        vector=embedding.tolist(),
        top_k=1,
        include_metadata=True
    )
    if not results.matches:
        return
    match = results.matches[0]
    similarity = match.score
    if similarity < 0.7:
        return

    person_id = match.metadata["person_id"]
    doc = db.collection("users").document(person_id).get()
    data = doc.to_dict() if doc.exists else {}
    person_obj = Person(
        bbox=bbox,
        person_id=person_id,
        height=data.get("height",185),
        weights=data.get("weights",{}),
        identified=True
    )

    person_obj.registration_bbox = bbox
    adapter.update_person(person_obj, frame, bbox)
    tracked_persons[person_id] = person_obj


    # gait thread
    person_obj.gait_queue = queue.Queue(maxsize=4)
    t = threading.Thread(
        target=run_gait_daemon,
        args=(person_obj,),
        daemon=True
    )
    t.start()
    person_obj.gait_thread = t


    # rPPG thread
    person_obj.rppg_queue = queue.Queue(maxsize=30)
    rppg_t = threading.Thread(
        target=run_rppg_daemon,
        args=(person_obj,),
        daemon=True
    )
    rppg_t.start()
    person_obj.rppg_thread = rppg_t


# ----------------------------
# gait daemon
# ----------------------------
def run_gait_daemon(person):
    while True:
        try:
            frame, bbox = person.gait_queue.get(timeout=5)
        except queue.Empty:
            continue
        adapter.update_person(person, frame, bbox)
        person.gait_queue.task_done()


# ----------------------------
# enqueue gait
# ----------------------------
def _enqueue_frame(person, frame, bbox):
    if hasattr(person, "gait_queue"):
        try:
            person.gait_queue.put_nowait((frame.copy(),bbox))
        except queue.Full:
            pass


# ----------------------------
# bbox matching
# ----------------------------
def match_bbox_to_person(bbox):
    x1,y1,x2,y2 = bbox
    center = np.array([(x1+x2)/2,(y1+y2)/2])
    for person in tracked_persons.values():
        if person.last_120_frames:
            ref_center = person.last_120_frames[-1].bbox_center
        elif hasattr(person,"registration_bbox"):
            rx1,ry1,rx2,ry2 = person.registration_bbox
            ref_center = np.array([(rx1+rx2)/2,(ry1+ry2)/2])
        else:
            continue
        if np.linalg.norm(center-ref_center) < MAX_DIST:
            return person

    return None


# ----------------------------
# CCTV loop
# ----------------------------
def cctv_loop():
    global raw_frame, overlay_frame
    cap = cv2.VideoCapture(0)
    with processor:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            raw_frame = frame.copy()
            results = yolo_model(frame)
            for box in results[0].boxes.xyxy.cpu().numpy().astype(int):
                bbox = tuple(box)
                person = match_bbox_to_person(bbox)
                if person is None:
                    temp = {"frame":frame.copy(),"bbox":bbox}
                    threading.Thread(
                        target=run_id_daemon,
                        args=(temp,),
                        daemon=True
                    ).start()

                else:
                    _enqueue_frame(person,frame,bbox)
                    _enqueue_rppg_frame(person,frame,bbox)

            overlay_frame = frame.copy()
            for p in tracked_persons.values():
                x1,y1,x2,y2 = p.bbox
                cv2.rectangle(
                    overlay_frame,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    2
                )

                label = f"{p.person_id}"
                if hasattr(p,"heart_rate"):
                    label += f" | HR {int(p.heart_rate)}"
                cv2.putText(
                    overlay_frame,
                    label,
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

            time.sleep(0.03)


# ----------------------------
# streaming
# ----------------------------
def gen_frame(mode="raw"):
    global raw_frame, overlay_frame

    while True:
        frame = raw_frame if mode == "raw" else overlay_frame

        if frame is None:
            time.sleep(0.01)
            continue

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n"
            b"\r\n" +
            frame_bytes +
            b"\r\n"
        )

        time.sleep(0.03)
# ----------------------------
# API routes
# ----------------------------
# ----------------------------
# WebSocket streaming
# ----------------------------

@router.websocket("/cctv/{cctv_id}/ws")
async def websocket_stream(websocket: WebSocket, cctv_id: int):
    await websocket.accept()

    try:
        while True:
            frame = overlay_frame

            if frame is None:
                await asyncio.sleep(0.01)
                continue

            ret, buffer = cv2.imencode(".jpg", frame)

            if not ret:
                continue

            await websocket.send_bytes(buffer.tobytes())

            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        print("Client disconnected")

@router.get("/cctv/{cctv_id}/persons")
def get_persons(cctv_id:int):
    return JSONResponse({
        "persons":[vars(p) for p in tracked_persons.values()]
    })


# ----------------------------
# startup
# ----------------------------
def start_cctv():
    t = threading.Thread(
        target=cctv_loop,
        daemon=True
    )
    t.start()

    print("CCTV daemon running")