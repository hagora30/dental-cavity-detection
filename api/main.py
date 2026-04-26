from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request
from pydantic import BaseModel
from typing import List
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

app = FastAPI(title="Dental Cavity Detection API")


class Detection(BaseModel):
    label: str
    confidence: float
    box_coordinates: List[float]

class PredictionResponse(BaseModel):
    detections: List[Detection]

try:
    model = YOLO("best.pt")
    logger.info("model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model weights. Error: {e}")
    model = None

CLASS_NAMES = ["Cavity", "Fillings", "Impacted Tooth", "Implant"]

@app.get("/")
def health_check():
    model_status = "active" if model else "offline"
    return {"status": model_status, "model": "yolov8m-dental-imgsz1280"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model is currently offline or failed to load"
        )

    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No file was uploaded"
        )

    content_length = request.headers.get('content-length')
    if content_length:
        if int(content_length) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
            )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, 
            detail="Invalid file format. Please upload an image (e.g., JPEG, PNG)"
        )

    try:
        image_bytes = await file.read()

        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB"
            )

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                detail="The uploaded file is corrupted or not a valid image"
            )
        try:
            results = model.predict(
                source=image,
                imgsz=1280,
                conf=0.30,
                iou=0.45,
                device="cpu",
                verbose=False
            )
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="An error occurred while processing the image"
            )

        # Process Results
        detections_list = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "Unknown"
                
                detections_list.append({
                    "label": cls_name,
                    "confidence": round(float(box.conf[0]), 2),
                    "box_coordinates": box.xyxy[0].tolist()
                })
                
        return {"detections": detections_list}

    except HTTPException:
        raise 
    except Exception as e:
        logger.error(f"Unexpected API error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="An unexpected server error occurred"
        )