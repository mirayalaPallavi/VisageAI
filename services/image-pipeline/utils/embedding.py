import numpy as np
from typing import List, Optional, Dict, Any
import structlog
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

logger = structlog.get_logger()

class EmbeddingService:
    """Service for generating face embeddings"""
    
    def __init__(self):
        self.embedding_service_url = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8006")
        self.timeout = 30.0
        
    async def generate_embeddings(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate embeddings for face images
        
        Args:
            face_images: List of preprocessed face images
            
        Returns:
            List of embeddings
        """
        try:
            # Convert numpy arrays to base64 strings for HTTP transmission
            encoded_images = self._encode_images(face_images)
            
            # Send to embedding service
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.embedding_service_url}/embed",
                    json={"images": encoded_images},
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embeddings = self._decode_embeddings(result["embeddings"])
                    logger.info(f"Generated {len(embeddings)} embeddings successfully")
                    return embeddings
                else:
                    logger.error(f"Embedding service error: {response.status_code} - {response.text}")
                    raise Exception(f"Embedding service error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
    
    def _encode_images(self, images: List[np.ndarray]) -> List[str]:
        """Encode numpy arrays to base64 strings"""
        import base64
        import io
        from PIL import Image
        
        encoded = []
        for img in images:
            # Convert to PIL Image
            if len(img.shape) == 3:
                pil_img = Image.fromarray((img * 255).astype(np.uint8))
            else:
                pil_img = Image.fromarray(img.astype(np.uint8))
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            encoded.append(img_str)
        
        return encoded
    
    def _decode_embeddings(self, encoded_embeddings: List[str]) -> List[np.ndarray]:
        """Decode base64 strings back to numpy arrays"""
        import base64
        
        embeddings = []
        for encoded in encoded_embeddings:
            # Decode base64
            decoded = base64.b64decode(encoded)
            
            # Convert to numpy array
            embedding = np.frombuffer(decoded, dtype=np.float32)
            embeddings.append(embedding)
        
        return embeddings
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, method: str = "cosine") -> float:
        """
        Calculate similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            method: Similarity method ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            Similarity score
        """
        if method == "cosine":
            return self._cosine_similarity(embedding1, embedding2)
        elif method == "euclidean":
            return self._euclidean_similarity(embedding1, embedding2)
        elif method == "manhattan":
            return self._manhattan_similarity(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _euclidean_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Euclidean distance similarity (1 / (1 + distance))"""
        distance = np.linalg.norm(a - b)
        return 1.0 / (1.0 + distance)
    
    def _manhattan_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Manhattan distance similarity (1 / (1 + distance))"""
        distance = np.sum(np.abs(a - b))
        return 1.0 / (1.0 + distance)
    
    def find_similar_faces(self, query_embedding: np.ndarray, gallery_embeddings: List[np.ndarray], 
                          threshold: float = 0.7, method: str = "cosine") -> List[Dict[str, Any]]:
        """
        Find similar faces in a gallery
        
        Args:
            query_embedding: Query face embedding
            gallery_embeddings: List of gallery embeddings
            threshold: Similarity threshold
            method: Similarity method
            
        Returns:
            List of similar faces with indices and scores
        """
        similar_faces = []
        
        for i, gallery_embedding in enumerate(gallery_embeddings):
            similarity = self.calculate_similarity(query_embedding, gallery_embedding, method)
            
            if similarity >= threshold:
                similar_faces.append({
                    "index": i,
                    "similarity": similarity,
                    "method": method
                })
        
        # Sort by similarity score (descending)
        similar_faces.sort(key=lambda x: x["similarity"], reverse=True)
        
        logger.info(f"Found {len(similar_faces)} similar faces above threshold {threshold}")
        return similar_faces
    
    def batch_similarity_search(self, query_embeddings: List[np.ndarray], 
                               gallery_embeddings: List[np.ndarray], 
                               threshold: float = 0.7, method: str = "cosine") -> List[List[Dict[str, Any]]]:
        """
        Perform batch similarity search
        
        Args:
            query_embeddings: List of query embeddings
            gallery_embeddings: List of gallery embeddings
            threshold: Similarity threshold
            method: Similarity method
            
        Returns:
            List of similar faces for each query
        """
        results = []
        
        for query_embedding in query_embeddings:
            similar_faces = self.find_similar_faces(
                query_embedding, gallery_embeddings, threshold, method
            )
            results.append(similar_faces)
        
        return results
    
    def validate_embedding(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Validate embedding quality
        
        Args:
            embedding: Face embedding
            
        Returns:
            Validation results
        """
        validation = {
            "is_valid": True,
            "dimension": len(embedding),
            "has_nan": np.any(np.isnan(embedding)),
            "has_inf": np.any(np.isinf(embedding)),
            "norm": float(np.linalg.norm(embedding)),
            "mean": float(np.mean(embedding)),
            "std": float(np.std(embedding))
        }
        
        # Check for common issues
        if validation["has_nan"] or validation["has_inf"]:
            validation["is_valid"] = False
            validation["issues"] = ["Contains NaN or Inf values"]
        
        if validation["norm"] == 0:
            validation["is_valid"] = False
            validation["issues"] = ["Zero vector embedding"]
        
        if validation["std"] < 1e-6:
            validation["warnings"] = ["Very low variance in embedding"]
        
        return validation
