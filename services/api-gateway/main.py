from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import httpx
import structlog
from typing import Dict, Any
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

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

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')

app = FastAPI(
    title="Club Project API Gateway",
    description="API Gateway for Club Project microservices",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure based on your needs
)

# Service URLs (configure via environment variables)
SERVICES = {
    "auth": "http://auth:8001",
    "template": "http://template-service:8002",
    "image": "http://image-pipeline:8003",
    "video": "http://video-pipeline:8004",
    "embedding": "http://embedding-service:8006",
    "matching": "http://matching-service:8007"
}

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None
    )
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time
    )
    
    # Update metrics
    REQUEST_COUNT.labels(method=request.method, endpoint=str(request.url.path), status=response.status_code).inc()
    REQUEST_LATENCY.observe(process_time)
    
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "api-gateway"}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    # Check if all services are reachable
    service_status = {}
    async with httpx.AsyncClient() as client:
        for service_name, service_url in SERVICES.items():
            try:
                response = await client.get(f"{service_url}/health", timeout=5.0)
                service_status[service_name] = "healthy" if response.status_code == 200 else "unhealthy"
            except Exception as e:
                logger.warning(f"Service {service_name} health check failed", error=str(e))
                service_status[service_name] = "unreachable"
    
    all_healthy = all(status == "healthy" for status in service_status.values())
    return {
        "status": "ready" if all_healthy else "not_ready",
        "services": service_status
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Club Project API Gateway",
        "version": "1.0.0",
        "status": "running"
    }

# Route forwarding to microservices
@app.api_route("/auth/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def auth_proxy(request: Request, path: str):
    """Proxy requests to auth service"""
    return await proxy_request(request, "auth", path)

@app.api_route("/templates/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def template_proxy(request: Request, path: str):
    """Proxy requests to template service"""
    return await proxy_request(request, "template", path)

@app.api_route("/images/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def image_proxy(request: Request, path: str):
    """Proxy requests to image pipeline service"""
    return await proxy_request(request, "image", path)

@app.api_route("/videos/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def video_proxy(request: Request, path: str):
    """Proxy requests to video pipeline service"""
    return await proxy_request(request, "video", path)

async def proxy_request(request: Request, service: str, path: str):
    """Generic proxy function to forward requests to microservices"""
    if service not in SERVICES:
        raise HTTPException(status_code=404, detail=f"Service {service} not found")
    
    service_url = SERVICES[service]
    target_url = f"{service_url}/{path}"
    
    # Get request body
    body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.body()
        except Exception:
            pass
    
    # Get query parameters
    query_params = dict(request.query_params)
    
    # Forward headers (excluding some that shouldn't be forwarded)
    headers = dict(request.headers)
    headers_to_remove = ["host", "content-length", "transfer-encoding"]
    for header in headers_to_remove:
        headers.pop(header, None)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                params=query_params,
                headers=headers,
                content=body,
                timeout=30.0
            )
            
            # Return response from microservice
            return JSONResponse(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
            
    except httpx.RequestError as e:
        logger.error(f"Request to {service} failed", error=str(e), service=service, path=path)
        raise HTTPException(status_code=503, detail=f"Service {service} unavailable")
    except Exception as e:
        logger.error(f"Unexpected error proxying to {service}", error=str(e), service=service, path=path)
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
