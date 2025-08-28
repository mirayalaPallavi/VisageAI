from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import structlog
import numpy as np
import base64
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

# Import vector search engines
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

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
    title="Club Project Matching Service",
    description="Service for vector similarity search and face matching",
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

# Global variables for search engines
faiss_index = None
milvus_collection = None
milvus_connected = False

class VectorSearchEngine:
    """Abstract base class for vector search engines"""
    
    def __init__(self):
        self.engine_type = "unknown"
    
    def add_vectors(self, vectors: List[np.ndarray], ids: List[str]) -> bool:
        """Add vectors to the search index"""
        raise NotImplementedError
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        raise NotImplementedError
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from the search index"""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        raise NotImplementedError

class FAISSSearchEngine(VectorSearchEngine):
    """FAISS-based vector search engine"""
    
    def __init__(self, dimension: int = 512):
        super().__init__()
        self.engine_type = "faiss"
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.vector_ids = []
        self.vector_map = {}  # id -> index mapping
    
    def add_vectors(self, vectors: List[np.ndarray], ids: List[str]) -> bool:
        """Add vectors to FAISS index"""
        try:
            if len(vectors) != len(ids):
                raise ValueError("Number of vectors must match number of IDs")
            
            # Convert to float32 if needed
            vectors_array = np.array(vectors, dtype=np.float32)
            
            # Add to index
            start_index = len(self.vector_ids)
            self.index.add(vectors_array)
            
            # Store IDs and mapping
            for i, vector_id in enumerate(ids):
                self.vector_ids.append(vector_id)
                self.vector_map[vector_id] = start_index + i
            
            logger.info(f"Added {len(vectors)} vectors to FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vectors to FAISS: {str(e)}")
            return False
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors using FAISS"""
        try:
            # Ensure query vector is float32
            query_vector = query_vector.astype(np.float32).reshape(1, -1)
            
            # Search
            similarities, indices = self.index.search(query_vector, min(k, len(self.vector_ids)))
            
            results = []
            for i, (similarity, index) in enumerate(zip(similarities[0], indices[0])):
                if index < len(self.vector_ids):
                    results.append({
                        "rank": i + 1,
                        "id": self.vector_ids[index],
                        "similarity": float(similarity),
                        "index": int(index)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {str(e)}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from FAISS index (not supported in basic FAISS)"""
        logger.warning("Vector deletion not supported in basic FAISS index")
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics"""
        return {
            "engine_type": self.engine_type,
            "total_vectors": len(self.vector_ids),
            "dimension": self.dimension,
            "index_type": "IndexFlatIP"
        }

class MilvusSearchEngine(VectorSearchEngine):
    """Milvus-based vector search engine"""
    
    def __init__(self, collection_name: str = "face_embeddings", dimension: int = 512):
        super().__init__()
        self.engine_type = "milvus"
        self.collection_name = collection_name
        self.dimension = dimension
        self.collection = None
        self.connected = False
    
    def connect(self, host: str = "localhost", port: int = 19530) -> bool:
        """Connect to Milvus server"""
        try:
            connections.connect(host=host, port=port)
            self.connected = True
            logger.info(f"Connected to Milvus at {host}:{port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            return False
    
    def create_collection(self) -> bool:
        """Create Milvus collection for face embeddings"""
        try:
            if not self.connected:
                raise Exception("Not connected to Milvus")
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
            ]
            schema = CollectionSchema(fields=fields, description="Face embeddings collection")
            
            # Create collection
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
            
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            # Create index
            index_params = {
                "metric_type": "IP",  # Inner product for cosine similarity
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            
            logger.info(f"Created Milvus collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Milvus collection: {str(e)}")
            return False
    
    def load_collection(self) -> bool:
        """Load existing Milvus collection"""
        try:
            if not self.connected:
                raise Exception("Not connected to Milvus")
            
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
                logger.info(f"Loaded Milvus collection: {self.collection_name}")
                return True
            else:
                logger.warning(f"Collection {self.collection_name} does not exist")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load Milvus collection: {str(e)}")
            return False
    
    def add_vectors(self, vectors: List[np.ndarray], ids: List[str]) -> bool:
        """Add vectors to Milvus collection"""
        try:
            if not self.collection:
                raise Exception("Collection not loaded")
            
            if len(vectors) != len(ids):
                raise ValueError("Number of vectors must match number of IDs")
            
            # Prepare data
            data = [
                ids,
                [vector.tolist() for vector in vectors]
            ]
            
            # Insert data
            self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"Added {len(vectors)} vectors to Milvus collection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vectors to Milvus: {str(e)}")
            return False
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors using Milvus"""
        try:
            if not self.collection:
                raise Exception("Collection not loaded")
            
            # Search parameters
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10}
            }
            
            # Perform search
            results = self.collection.search(
                data=[query_vector.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=k,
                output_fields=["id"]
            )
            
            # Format results
            formatted_results = []
            for i, hit in enumerate(results[0]):
                formatted_results.append({
                    "rank": i + 1,
                    "id": hit.entity.get("id"),
                    "similarity": float(hit.score),
                    "distance": float(hit.distance)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Milvus search failed: {str(e)}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from Milvus collection"""
        try:
            if not self.collection:
                raise Exception("Collection not loaded")
            
            # Delete by IDs
            expr = f'id in {ids}'
            self.collection.delete(expr)
            
            logger.info(f"Deleted {len(ids)} vectors from Milvus collection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors from Milvus: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Milvus collection statistics"""
        try:
            if not self.collection:
                return {"engine_type": self.engine_type, "status": "not_loaded"}
            
            stats = self.collection.get_statistics()
            return {
                "engine_type": self.engine_type,
                "collection_name": self.collection_name,
                "total_vectors": stats["row_count"],
                "dimension": self.dimension,
                "status": "loaded"
            }
            
        except Exception as e:
            logger.error(f"Failed to get Milvus stats: {str(e)}")
            return {"engine_type": self.engine_type, "status": "error"}

def initialize_search_engines():
    """Initialize vector search engines"""
    global faiss_index, milvus_collection, milvus_connected
    
    try:
        # Initialize FAISS
        if FAISS_AVAILABLE:
            faiss_index = FAISSSearchEngine(dimension=512)
            logger.info("FAISS search engine initialized")
        else:
            logger.warning("FAISS not available")
        
        # Initialize Milvus
        if MILVUS_AVAILABLE:
            milvus_host = os.getenv("MILVUS_HOST", "localhost")
            milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
            
            milvus_collection = MilvusSearchEngine(collection_name="face_embeddings", dimension=512)
            milvus_connected = milvus_collection.connect(milvus_host, milvus_port)
            
            if milvus_connected:
                # Try to load existing collection, create if not exists
                if not milvus_collection.load_collection():
                    milvus_collection.create_collection()
                logger.info("Milvus search engine initialized")
            else:
                logger.warning("Failed to connect to Milvus")
        else:
            logger.warning("Milvus not available")
            
    except Exception as e:
        logger.error(f"Failed to initialize search engines: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize search engines on startup"""
    initialize_search_engines()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    faiss_status = "available" if faiss_index else "not_available"
    milvus_status = "connected" if milvus_connected else "not_connected"
    
    return {
        "status": "healthy",
        "service": "matching-service",
        "faiss": faiss_status,
        "milvus": milvus_status
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    if not faiss_index and not milvus_connected:
        return {
            "status": "not_ready",
            "service": "matching-service",
            "error": "No search engines available"
        }
    
    return {"status": "ready", "service": "matching-service"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Club Project Matching Service",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/add-vectors")
async def add_vectors(request: Dict[str, Any]):
    """
    Add vectors to the search index
    
    Args:
        request: Dictionary with vectors and IDs
        
    Returns:
        Addition status
    """
    try:
        vectors_b64 = request.get("vectors", [])
        ids = request.get("ids", [])
        
        if not vectors_b64 or not ids:
            raise HTTPException(status_code=400, detail="Vectors and IDs are required")
        
        if len(vectors_b64) != len(ids):
            raise HTTPException(status_code=400, detail="Number of vectors must match number of IDs")
        
        # Decode base64 vectors
        vectors = []
        for vector_b64 in vectors_b64:
            vector_bytes = base64.b64decode(vector_b64)
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            vectors.append(vector)
        
        # Add to search engines
        results = {}
        
        if faiss_index:
            faiss_success = faiss_index.add_vectors(vectors, ids)
            results["faiss"] = "success" if faiss_success else "failed"
        
        if milvus_collection and milvus_connected:
            milvus_success = milvus_collection.add_vectors(vectors, ids)
            results["milvus"] = "success" if milvus_success else "failed"
        
        logger.info(f"Added {len(vectors)} vectors to search engines", results=results)
        
        return {
            "message": f"Added {len(vectors)} vectors",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Failed to add vectors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add vectors: {str(e)}")

@app.post("/search")
async def search_vectors(request: Dict[str, Any]):
    """
    Search for similar vectors
    
    Args:
        request: Dictionary with query vector and search parameters
        
    Returns:
        Search results
    """
    try:
        query_vector_b64 = request.get("query_vector")
        k = request.get("k", 10)
        engine = request.get("engine", "auto")  # auto, faiss, or milvus
        
        if not query_vector_b64:
            raise HTTPException(status_code=400, detail="Query vector is required")
        
        # Decode query vector
        query_vector_bytes = base64.b64decode(query_vector_b64)
        query_vector = np.frombuffer(query_vector_bytes, dtype=np.float32)
        
        # Determine which engine to use
        if engine == "faiss" and faiss_index:
            search_engine = faiss_index
        elif engine == "milvus" and milvus_collection and milvus_connected:
            search_engine = milvus_collection
        elif engine == "auto":
            # Use FAISS if available, otherwise Milvus
            if faiss_index:
                search_engine = faiss_index
            elif milvus_collection and milvus_connected:
                search_engine = milvus_collection
            else:
                raise HTTPException(status_code=503, detail="No search engines available")
        else:
            raise HTTPException(status_code=400, detail=f"Engine {engine} not available")
        
        # Perform search
        results = search_engine.search(query_vector, k)
        
        logger.info(f"Search completed", engine=search_engine.engine_type, results_count=len(results))
        
        return {
            "query_vector_dimension": len(query_vector),
            "k": k,
            "engine_used": search_engine.engine_type,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.delete("/delete-vectors")
async def delete_vectors(request: Dict[str, Any]):
    """
    Delete vectors from the search index
    
    Args:
        request: Dictionary with vector IDs to delete
        
    Returns:
        Deletion status
    """
    try:
        ids = request.get("ids", [])
        
        if not ids:
            raise HTTPException(status_code=400, detail="Vector IDs are required")
        
        # Delete from search engines
        results = {}
        
        if faiss_index:
            faiss_success = faiss_index.delete_vectors(ids)
            results["faiss"] = "success" if faiss_success else "failed"
        
        if milvus_collection and milvus_connected:
            milvus_success = milvus_collection.delete_vectors(ids)
            results["milvus"] = "success" if milvus_success else "failed"
        
        logger.info(f"Deleted {len(ids)} vectors from search engines", results=results)
        
        return {
            "message": f"Deleted {len(ids)} vectors",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Failed to delete vectors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete vectors: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get search engine statistics"""
    try:
        stats = {}
        
        if faiss_index:
            stats["faiss"] = faiss_index.get_stats()
        
        if milvus_collection:
            stats["milvus"] = milvus_collection.get_stats()
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
