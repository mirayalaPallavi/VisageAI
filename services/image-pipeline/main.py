from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
import cv2
import numpy as np
from typing import List, Optional
import io
from PIL import Image
import os
from dotenv import load_dotenv

from utils.detection import FaceDetector
from utils.embedding import EmbeddingService

# Load environment variables
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

app = FastAPI(
    title="Club Project Image Pipeline",
    description="Image processing pipeline for face detection and embedding",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
face_detector = FaceDetector()
embedding_service = EmbeddingService()

def image_to_numpy(image_bytes: bytes) -> np.ndarray:
    """Convert image bytes to numpy array"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Convert RGBA to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        return image_array
    except Exception as e:
        logger.error(f"Failed to convert image to numpy array: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image format")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "image-pipeline"}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        # Check if face detector is working
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = face_detector.detect_faces(test_image)
        
        return {"status": "ready", "service": "image-pipeline"}
    except Exception as e:
        logger.error("Service not ready", error=str(e))
        return {"status": "not_ready", "service": "image-pipeline", "error": str(e)}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Club Project Image Pipeline",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/detect-faces")
async def detect_faces(
    image: UploadFile = File(...),
    min_face_size: int = Form(30),
    scale_factor: float = Form(1.1),
    min_neighbors: int = Form(5)
):
    """
    Detect faces in an uploaded image
    
    Args:
        image: Image file to process
        min_face_size: Minimum face size in pixels
        scale_factor: Scale factor for detection
        min_neighbors: Minimum neighbors for detection
        
    Returns:
        List of detected face bounding boxes and quality metrics
    """
    try:
        # Read and validate image
        image_bytes = await image.read()
        image_array = image_to_numpy(image_bytes)
        
        logger.info(f"Processing image: {image.filename}, size: {image_array.shape}")
        
        # Detect faces
        faces = face_detector.detect_faces(image_array)
        
        # Extract face regions and analyze quality
        face_results = []
        for i, bbox in enumerate(faces):
            # Extract face region
            face_region = face_detector.extract_face_region(image_array, bbox)
            
            # Analyze quality
            quality_metrics = face_detector.validate_face_quality(face_region)
            
            face_results.append({
                "face_id": i,
                "bbox": bbox,
                "quality_metrics": quality_metrics,
                "face_size": face_region.shape
            })
        
        logger.info(f"Detected {len(faces)} faces in image")
        
        return {
            "image_info": {
                "filename": image.filename,
                "size": image_array.shape,
                "total_faces": len(faces)
            },
            "faces": face_results
        }
        
    except Exception as e:
        logger.error(f"Face detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Face detection failed: {str(e)}")

@app.post("/extract-faces")
async def extract_faces(
    image: UploadFile = File(...),
    target_size: int = Form(224),
    quality_threshold: int = Form(50)
):
    """
    Extract and preprocess faces from an image
    
    Args:
        image: Image file to process
        target_size: Target size for extracted faces
        quality_threshold: Minimum quality score for faces
        
    Returns:
        List of extracted faces with metadata
    """
    try:
        # Read and validate image
        image_bytes = await image.read()
        image_array = image_to_numpy(image_bytes)
        
        logger.info(f"Extracting faces from image: {image.filename}")
        
        # Detect and extract faces
        target_size_tuple = (target_size, target_size)
        extracted_faces = face_detector.detect_and_extract_faces(image_array, target_size_tuple)
        
        # Filter by quality
        quality_faces = []
        for i, face in enumerate(extracted_faces):
            quality_metrics = face_detector.validate_face_quality(face)
            
            if quality_metrics["quality_score"] >= quality_threshold:
                quality_faces.append({
                    "face_id": i,
                    "face_size": face.shape,
                    "quality_metrics": quality_metrics,
                    "preprocessed": True
                })
        
        logger.info(f"Extracted {len(quality_faces)} quality faces from {len(extracted_faces)} total faces")
        
        return {
            "image_info": {
                "filename": image.filename,
                "size": image_array.shape
            },
            "total_faces_detected": len(extracted_faces),
            "quality_faces": quality_faces,
            "quality_threshold": quality_threshold
        }
        
    except Exception as e:
        logger.error(f"Face extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Face extraction failed: {str(e)}")

@app.post("/generate-embeddings")
async def generate_embeddings(
    image: UploadFile = File(...),
    target_size: int = Form(224),
    quality_threshold: int = Form(50)
):
    """
    Generate embeddings for faces in an image
    
    Args:
        image: Image file to process
        target_size: Target size for face preprocessing
        quality_threshold: Minimum quality score for faces
        
    Returns:
        List of face embeddings with metadata
    """
    try:
        # Read and validate image
        image_bytes = await image.read()
        image_array = image_to_numpy(image_bytes)
        
        logger.info(f"Generating embeddings for image: {image.filename}")
        
        # Extract faces
        target_size_tuple = (target_size, target_size)
        extracted_faces = face_detector.detect_and_extract_faces(image_array, target_size_tuple)
        
        if not extracted_faces:
            return {
                "image_info": {
                    "filename": image.filename,
                    "size": image_array.shape
                },
                "message": "No faces detected in image",
                "embeddings": []
            }
        
        # Generate embeddings
        embeddings = await embedding_service.generate_embeddings(extracted_faces)
        
        # Validate embeddings
        embedding_results = []
        for i, (face, embedding) in enumerate(zip(extracted_faces, embeddings)):
            validation = embedding_service.validate_embedding(embedding)
            
            embedding_results.append({
                "face_id": i,
                "embedding_dimension": len(embedding),
                "embedding_validation": validation,
                "face_size": face.shape
            })
        
        logger.info(f"Generated {len(embeddings)} embeddings successfully")
        
        return {
            "image_info": {
                "filename": image.filename,
                "size": image_array.shape
            },
            "total_faces": len(extracted_faces),
            "embeddings": embedding_results
        }
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.post("/process-image")
async def process_image(
    image: UploadFile = File(...),
    target_size: int = Form(224),
    quality_threshold: int = Form(50),
    generate_embeddings_flag: bool = Form(True)
):
    """
    Complete image processing pipeline
    
    Args:
        image: Image file to process
        target_size: Target size for face preprocessing
        quality_threshold: Minimum quality score for faces
        generate_embeddings_flag: Whether to generate embeddings
        
    Returns:
        Complete processing results
    """
    try:
        # Read and validate image
        image_bytes = await image.read()
        image_array = image_to_numpy(image_bytes)
        
        logger.info(f"Processing image: {image.filename}")
        
        # Step 1: Detect faces
        faces = face_detector.detect_faces(image_array)
        
        if not faces:
            return {
                "image_info": {
                    "filename": image.filename,
                    "size": image_array.shape
                },
                "message": "No faces detected in image",
                "processing_steps": ["face_detection"],
                "results": {
                    "faces_detected": 0,
                    "faces_extracted": 0,
                    "embeddings_generated": 0
                }
            }
        
        # Step 2: Extract and preprocess faces
        target_size_tuple = (target_size, target_size)
        extracted_faces = face_detector.detect_and_extract_faces(image_array, target_size_tuple)
        
        # Step 3: Generate embeddings if requested
        embeddings = []
        if generate_embeddings_flag and extracted_faces:
            embeddings = await embedding_service.generate_embeddings(extracted_faces)
        
        # Compile results
        processing_steps = ["face_detection", "face_extraction"]
        if generate_embeddings_flag:
            processing_steps.append("embedding_generation")
        
        results = {
            "faces_detected": len(faces),
            "faces_extracted": len(extracted_faces),
            "embeddings_generated": len(embeddings),
            "processing_steps": processing_steps
        }
        
        logger.info(f"Image processing completed: {results}")
        
        return {
            "image_info": {
                "filename": image.filename,
                "size": image_array.shape
            },
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
