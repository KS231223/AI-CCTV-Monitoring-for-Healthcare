import cv2
import threading
import numpy as np
from ultralytics import YOLO
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from entities import Person
from frame_processor import DecoupledProcessor, PersonAdapter
import firebase_admin
from firebase_admin import credentials, firestore
from pinecone import Pinecone, ServerlessSpec
from rPPGmicroservice import rppg_simple

# ----------------------------
# Firebase setup
# ----------------------------
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ----------------------------
# Pinecone setup (v3+ SDK)
# ----------------------------
pc = Pinecone(api_key="")
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
# Models
# ----------------------------
yolo_model = YOLO("yolo26n.pt")
mtcnn = MTCNN(keep_all=True)
face_model = InceptionResnetV1(pretrained="vggface2").eval()

processor = DecoupledProcessor(model_path="pose_landmarker_heavy.task", fps=30)
adapter = PersonAdapter(processor)

# ----------------------------
# Global trackers
# ----------------------------
tracked_persons: dict[str, Person] = {}
MAX_DIST = 120  # px — max bbox-centre distance for identity persistence

# ----------------------------
# Embedding helper
# ----------------------------
def compute_embedding(face_crop: np.ndarray) -> np.ndarray:
    face = Image.fromarray(face_crop).resize((160, 160))
    tensor = torch.tensor(np.array(face).transpose(2, 0, 1)).float() / 255.0
    tensor = tensor.unsqueeze(0)
    emb = face_model(tensor).detach().numpy()[0]
    emb /= np.linalg.norm(emb) + 1e-8
    return emb

# ----------------------------
# ID processing (one-shot per new person)
# ----------------------------
def run_id_thread(frame: np.ndarray, bbox: tuple):
    x1, y1, x2, y2 = bbox
    person_crop = frame[y1:y2, x1:x2]

    boxes, _ = mtcnn.detect(person_crop)
    if boxes is None or len(boxes) == 0:
        return

    fx1, fy1, fx2, fy2 = map(int, boxes[0])
    face_crop = person_crop[fy1:fy2, fx1:fx2]
    if face_crop.size == 0:
        return

    embedding = compute_embedding(face_crop)
    results = index.query(vector=embedding.tolist(), top_k=1, include_metadata=True)
    if not results.matches:
        return

    match = results.matches[0]
    if match.score < 0.7:
        return

    person_id = match.metadata["person_id"]
    doc = db.collection("users").document(person_id).get()
    data = doc.to_dict() if doc.exists else {}

    person_obj = Person(
        bbox=bbox,
        person_id=person_id,
        height=data.get("height", 185),
        weights=data.get("weights", {}),
        identified=True,
        heart_rate = 0
    )
    
    person_obj.registration_bbox = bbox
    adapter.update_person(person_obj, frame, bbox)
    tracked_persons[person_id] = person_obj

# ----------------------------
# Gait per-frame processing
# ----------------------------
def process_gait_frame(person: Person, frame: np.ndarray, bbox: tuple):
    adapter.update_person(person, frame, bbox)

# ----------------------------
# rPPG per-frame processing
# ----------------------------
def process_rppg_frame(person: Person, frame: np.ndarray, bbox: tuple):
    print(person.heart_rate, "HELLO")
    x1, y1, x2, y2 = bbox
    person_crop = frame[y1:y2, x1:x2]
    boxes, _ = mtcnn.detect(person_crop)
    if boxes is None or len(boxes) == 0:
        return
    fx1, fy1, fx2, fy2 = map(int, boxes[0])

    roi = person_crop[fy1:int(fy1 + (fy2 - fy1) * 0.3), fx1:fx2]
    if roi.size == 0:
        return

    person.last_120_rPPG.append(roi)
    if len(person.last_120_rPPG) > 120:
        person.last_120_rPPG.pop(0)

    if len(person.last_120_rPPG) >= 60:
        hr, var = rppg_simple(person.last_120_rPPG)
        person.heart_rate = float(hr)
        person.rppg_variance = float(var)
        print(person.heart_rate)

# ----------------------------
# Bbox-to-person matching
# ----------------------------
def match_bbox_to_person(bbox: tuple) -> "Person | None":
    x1, y1, x2, y2 = bbox
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    for person in tracked_persons.values():
        if person.last_120_frames and person.last_120_frames[-1].bbox_center is not None:
            ref_center = person.last_120_frames[-1].bbox_center
        elif hasattr(person, "registration_bbox"):
            rx1, ry1, rx2, ry2 = person.registration_bbox
            ref_center = np.array([(rx1 + rx2) / 2, (ry1 + ry2) / 2])
        else:
            continue
        if np.linalg.norm(center - ref_center) < MAX_DIST:
            return person
    return None

# ----------------------------
# Main loop
# ----------------------------
def main():
    cap = cv2.VideoCapture(0)

    with processor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model(frame)
            for box in results[0].boxes.xyxy.cpu().numpy().astype(int):
                bbox = tuple(box)
                person = match_bbox_to_person(bbox)

                if person is None:
                    threading.Thread(target=run_id_thread, args=(frame.copy(), bbox), daemon=True).start()
                    label = "Unknown"
                else:
                    # Gait and rPPG as per-frame threads
                    threading.Thread(target=process_gait_frame, args=(person, frame.copy(), bbox), daemon=True).start()
                    threading.Thread(target=process_rppg_frame, args=(person, frame.copy(), bbox), daemon=True).start()
                    label = person.display_id +" " + str(person.heart_rate)

                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Re-ID + Gait + rPPG", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()