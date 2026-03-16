# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cctv_service import router as cctv_router
from cctv_service import start_cctv

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# root
@app.get("/")
def root():
    return {"message": "Backend running"}

# mount CCTV router
app.include_router(cctv_router)

# startup
@app.on_event("startup")
def startup_event():
    print("Starting CCTV")
    start_cctv()
    