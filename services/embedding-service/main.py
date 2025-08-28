from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import structlog
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

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
    title="Club Project Embedding Service",
    description="Service for generating face embeddings using deep learning models",
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

# Global model variable
embedding_model = None
model_device = None

class FaceEmbeddingModel(nn.Module):
    """Simple face embedding model using ResNet-like architecture"""
    
    def __init__(self, embedding_dim=512):
        super(FaceEmbeddingModel, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # L2 normalization
        self.l2_norm = lambda x: x / torch.norm(x, p=2, dim=1, keepdim=True)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = self.l2_norm(x)
        return x

def load_model():
    """Load the face embedding model"""
    global embedding_model, model_device
    
    try:
        # Set device
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {model_device}")
        
        # Initialize model
        embedding_model = FaceEmbeddingModel(embedding_dim=512)
        embedding_model.to(model_device)
        embedding_model.eval()
        
        # Load pre-trained weights if available
        model_path = os.getenv("MODEL_PATH", "/app/models/face_embedding_model.pth")
        if os.path.exists(model_path):
            embedding_model.load_state_dict(torch.load(model_path, map_location=model_device))
            logger.info(f"Loaded pre-trained model from {model_path}")
        else:
            logger.info("No pre-trained model found, using random weights")
        
        logger.info("Face embedding model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocess image for model input
    
    Args:
        image_bytes: Image bytes
        
    Returns:
        Preprocessed image tensor
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms
        image_tensor = transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image format")

def generate_embedding(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Generate embedding for preprocessed image
    
    Args:
        image_tensor: Preprocessed image tensor
        
    Returns:
        Face embedding as numpy array
    """
    try:
        if embedding_model is None:
            raise Exception("Model not loaded")
        
        # Move tensor to device
        image_tensor = image_tensor.to(model_device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = embedding_model(image_tensor)
        
        # Convert to numpy
        embedding_np = embedding.cpu().numpy()
        
        return embedding_np[0]  # Remove batch dimension
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if embedding_model is not None else "not_loaded"
    return {
        "status": "healthy",
        "service": "embedding-service",
        "model_status": model_status,
        "device": str(model_device) if model_device else "unknown"
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    if embedding_model is None:
        return {
            "status": "not_ready",
            "service": "embedding-service",
            "error": "Model not loaded"
        }
    
    return {"status": "ready", "service": "embedding-service"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Club Project Embedding Service",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/embed")
async def embed_image(image: UploadFile = File(...)):
    """
    Generate embedding for a single image
    
    Args:
        image: Image file to process
        
    Returns:
        Face embedding
    """
    try:
        # Read image
        image_bytes = await image.read()
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        
        # Generate embedding
        embedding = generate_embedding(image_tensor)
        
        # Encode embedding to base64 for JSON response
        embedding_bytes = embedding.tobytes()
        embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')
        
        logger.info(f"Generated embedding for image: {image.filename}")
        
        return {
            "image_filename": image.filename,
            "embedding_dimension": len(embedding),
            "embedding": embedding_b64,
            "embedding_type": "float32"
        }
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.post("/embed-batch")
async def embed_batch_images(images: List[UploadFile] = File(...)):
    """
    Generate embeddings for multiple images
    
    Args:
        images: List of image files to process
        
    Returns:
        List of face embeddings
    """
    try:
        embeddings = []
        
        for image in images:
            try:
                # Read image
                image_bytes = await image.read()
                
                # Preprocess image
                image_tensor = preprocess_image(image_bytes)
                
                # Generate embedding
                embedding = generate_embedding(image_tensor)
                
                # Encode embedding to base64
                embedding_bytes = embedding.tobytes()
                embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')
                
                embeddings.append({
                    "image_filename": image.filename,
                    "embedding_dimension": len(embedding),
                    "embedding": embedding_b64,
                    "embedding_type": "float32"
                })
                
            except Exception as e:
                logger.error(f"Failed to process image {image.filename}: {str(e)}")
                embeddings.append({
                    "image_filename": image.filename,
                    "error": str(e)
                })
        
        logger.info(f"Generated embeddings for {len(images)} images")
        
        return {
            "total_images": len(images),
            "successful_embeddings": len([e for e in embeddings if "error" not in e]),
            "failed_embeddings": len([e for e in embeddings if "error" in e]),
            "embeddings": embeddings
        }
        
    except Exception as e:
        logger.error(f"Batch embedding generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch embedding generation failed: {str(e)}")

@app.post("/embed-base64")
async def embed_base64_images(request: Dict[str, Any]):
    """
    Generate embeddings for base64 encoded images
    
    Args:
        request: Dictionary with list of base64 encoded images
        
    Returns:
        List of face embeddings
    """
    try:
        images = request.get("images", [])
        
        if not images:
            raise HTTPException(status_code=400, detail="No images provided")
        
        embeddings = []
        
        for i, image_b64 in enumerate(images):
            try:
                # Decode base64 image
                image_bytes = base64.b64decode(image_b64)
                
                # Preprocess image
                image_tensor = preprocess_image(image_bytes)
                
                # Generate embedding
                embedding = generate_embedding(image_tensor)
                
                # Encode embedding to base64
                embedding_bytes = embedding.tobytes()
                embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')
                
                embeddings.append({
                    "image_index": i,
                    "embedding_dimension": len(embedding),
                    "embedding": embedding_b64,
                    "embedding_type": "float32"
                })
                
            except Exception as e:
                logger.error(f"Failed to process image {i}: {str(e)}")
                embeddings.append({
                    "image_index": i,
                    "error": str(e)
                })
        
        logger.info(f"Generated embeddings for {len(images)} base64 images")
        
        return {
            "total_images": len(images),
            "successful_embeddings": len([e for e in embeddings if "error" not in e]),
            "failed_embeddings": len([e for e in embeddings if "error" in e]),
            "embeddings": embeddings
        }
        
    except Exception as e:
        logger.error(f"Base64 embedding generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Base64 embedding generation failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_info = {
        "model_type": type(embedding_model).__name__,
        "embedding_dimension": 512,
        "device": str(model_device),
        "total_parameters": sum(p.numel() for p in embedding_model.parameters()),
        "trainable_parameters": sum(p.numel() for p in embedding_model.parameters() if p.requires_grad)
    }
    
    return model_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
