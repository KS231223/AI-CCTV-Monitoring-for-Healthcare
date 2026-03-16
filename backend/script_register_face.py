# script_register_face.py
import uuid
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, MTCNN
from pinecone import Pinecone, ServerlessSpec
import os
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

# ------------------------------
# 1. Generate a unique UUID
# ------------------------------



# ------------------------------
# 1. Initialize Pinecone
# ------------------------------
db = firestore.client()
pc = Pinecone(
    api_key=""
)

index_name = "faceembeddings"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=512,  # FaceNet embedding size
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# ------------------------------
# 2. Load models
# ------------------------------

yolo_model = YOLO("yolo26n.pt")  # your detector

mtcnn = MTCNN(keep_all=True)

face_model = InceptionResnetV1(pretrained="vggface2").eval()

# ------------------------------
# 3. Load image
# ------------------------------

image_path = "my_face.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

annotated = image.copy()

# ------------------------------
# 4. YOLO detection
# ------------------------------

results = yolo_model(image_rgb)

if len(results[0].boxes) == 0:
    raise Exception("No detection from YOLO")

for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy().astype(int)):
    person_id = str(uuid.uuid4())  # unique for this person/photo
    x1, y1, x2, y2 = box

    # draw YOLO box
    cv2.rectangle(annotated, (x1,y1),(x2,y2),(255,0,0),2)
    cv2.putText(annotated,"YOLO",(x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

    person_crop = image_rgb[y1:y2, x1:x2]

    # ------------------------------
    # 5. MTCNN face detection
    # ------------------------------

    boxes, _ = mtcnn.detect(person_crop)

    if boxes is None:
        continue

    for j, fbox in enumerate(boxes):

        fx1, fy1, fx2, fy2 = map(int, fbox)

        # convert back to original image coords
        fx1 += x1
        fx2 += x1
        fy1 += y1
        fy2 += y1

        # draw MTCNN box
        cv2.rectangle(annotated,(fx1,fy1),(fx2,fy2),(0,255,0),2)
        cv2.putText(annotated,"MTCNN",(fx1,fy1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        face_crop = image_rgb[fy1:fy2, fx1:fx2]

        # ------------------------------
        # 6. Generate embedding
        # ------------------------------

        face = Image.fromarray(face_crop).resize((160, 160))
        tensor = torch.tensor(np.array(face).transpose(2, 0, 1)).float() / 255.0
        tensor = tensor.unsqueeze(0)
        embedding = face_model(tensor).detach().numpy()[0]
        embedding /= np.linalg.norm(emb) + 1e-8

        # ------------------------------
        # 7. Store in Pinecone
        # ------------------------------

        vector_id = f"{person_id}_{i}_{j}"

        metadata = {
            "person_id": person_id,
        }

        index.upsert([
            (vector_id, embedding.tolist(), metadata)
        ])

        print(f"Stored embedding {vector_id}")
        user_doc = {
            "weights": {},          # empty dictionary
            "person_id": person_id, # same uuid
            "height": 1.85          # estimated
        }
        db.collection("users").document(person_id).set(user_doc)

        print(f"Firestore document created for person_id {person_id}")

# ------------------------------
# 8. Save annotated image
# ------------------------------

output_path = "annotated_" + os.path.basename(image_path)

cv2.imwrite(output_path, annotated)

print(f"Annotated image saved to {output_path}")