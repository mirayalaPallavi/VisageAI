from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
import os
import tempfile
from typing import Optional
from dotenv import load_dotenv

from tasks import (
    process_video_frames,
    extract_faces_from_video,
    analyze_video_liveness,
    celery_app
)

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
    title="Club Project Video Pipeline",
    description="Video processing pipeline for face detection and liveness analysis",
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "video-pipeline"}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        # Check if Celery is working
        celery_app.control.inspect().active()
        return {"status": "ready", "service": "video-pipeline"}
    except Exception as e:
        logger.error("Service not ready", error=str(e))
        return {"status": "not_ready", "service": "video-pipeline", "error": str(e)}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Club Project Video Pipeline",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/submit-video-job")
async def submit_video_job(
    video: UploadFile = File(...),
    job_type: str = Form("frame_processing"),
    frame_interval: int = Form(1),
    background_tasks: BackgroundTasks = None
):
    """
    Submit a video processing job
    
    Args:
        video: Video file to process
        job_type: Type of processing job
        frame_interval: Extract every Nth frame
        background_tasks: FastAPI background tasks
        
    Returns:
        Job submission details
    """
    try:
        # Validate video file
        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
            raise HTTPException(status_code=400, detail="Unsupported video format")
        
        # Save video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as temp_file:
            content = await video.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Video job submitted", filename=video.filename, job_type=job_type)
        
        # Submit job based on type
        if job_type == "frame_processing":
            task = process_video_frames.delay(temp_file_path, frame_interval)
        elif job_type == "face_extraction":
            task = extract_faces_from_video.delay(temp_file_path)
        elif job_type == "liveness_analysis":
            task = analyze_video_liveness.delay(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown job type: {job_type}")
        
        # Clean up temporary file after job submission
        background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        return {
            "job_id": task.id,
            "job_type": job_type,
            "status": "submitted",
            "message": f"Video processing job submitted successfully. Job ID: {task.id}"
        }
        
    except Exception as e:
        logger.error(f"Video job submission failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Job submission failed: {str(e)}")

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a video processing job
    
    Args:
        job_id: Celery task ID
        
    Returns:
        Job status and results
    """
    try:
        task = celery_app.AsyncResult(job_id)
        
        if task.state == 'PENDING':
            response = {
                'job_id': job_id,
                'state': task.state,
                'status': 'Job is waiting for execution or unknown'
            }
        elif task.state == 'PROGRESS':
            response = {
                'job_id': job_id,
                'state': task.state,
                'status': task.info.get('status', ''),
                'current': task.info.get('current', 0),
                'total': task.info.get('total', 0)
            }
        elif task.state == 'SUCCESS':
            response = {
                'job_id': job_id,
                'state': task.state,
                'status': 'Job completed successfully',
                'result': task.result
            }
        else:
            response = {
                'job_id': job_id,
                'state': task.state,
                'status': str(task.info)
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@app.post("/process-video-sync")
async def process_video_sync(
    video: UploadFile = File(...),
    frame_interval: int = Form(5),
    extract_faces: bool = Form(True),
    analyze_liveness: bool = Form(False)
):
    """
    Process video synchronously (for small videos)
    
    Args:
        video: Video file to process
        frame_interval: Extract every Nth frame
        extract_faces: Whether to extract faces
        analyze_liveness: Whether to analyze liveness
        
    Returns:
        Processing results
    """
    try:
        # Validate video file
        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
            raise HTTPException(status_code=400, detail="Unsupported video format")
        
        # Save video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as temp_file:
            content = await video.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            logger.info(f"Processing video synchronously", filename=video.filename)
            
            # Process video frames
            frame_task = process_video_frames.delay(temp_file_path, frame_interval)
            frame_results = frame_task.get()
            
            results = {
                "video_info": {
                    "filename": video.filename,
                    "total_frames": frame_results.get("total_frames", 0),
                    "processed_frames": frame_results.get("processed_frames", 0),
                    "fps": frame_results.get("fps", 0),
                    "duration": frame_results.get("duration", 0)
                },
                "frame_analysis": frame_results
            }
            
            # Extract faces if requested
            if extract_faces:
                face_task = extract_faces_from_video.delay(temp_file_path)
                face_results = face_task.get()
                results["face_extraction"] = face_results
            
            # Analyze liveness if requested
            if analyze_liveness:
                liveness_task = analyze_video_liveness.delay(temp_file_path)
                liveness_results = liveness_task.get()
                results["liveness_analysis"] = liveness_results
            
            return results
            
        finally:
            # Clean up temporary file
            cleanup_temp_file(temp_file_path)
        
    except Exception as e:
        logger.error(f"Synchronous video processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

@app.get("/active-jobs")
async def get_active_jobs():
    """Get list of active video processing jobs"""
    try:
        inspector = celery_app.control.inspect()
        active_jobs = inspector.active()
        
        if not active_jobs:
            return {"active_jobs": []}
        
        # Flatten the active jobs
        all_active_jobs = []
        for worker, jobs in active_jobs.items():
            for job in jobs:
                job_info = {
                    "job_id": job["id"],
                    "worker": worker,
                    "task": job["name"],
                    "start_time": job.get("time_start", 0),
                    "args": job.get("args", []),
                    "kwargs": job.get("kwargs", {})
                }
                all_active_jobs.append(job_info)
        
        return {"active_jobs": all_active_jobs}
        
    except Exception as e:
        logger.error(f"Failed to get active jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get active jobs: {str(e)}")

@app.delete("/cancel-job/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a video processing job
    
    Args:
        job_id: Celery task ID to cancel
        
    Returns:
        Cancellation status
    """
    try:
        celery_app.control.revoke(job_id, terminate=True)
        
        logger.info(f"Job cancelled", job_id=job_id)
        
        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": f"Job {job_id} has been cancelled"
        }
        
    except Exception as e:
        logger.error(f"Failed to cancel job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

def cleanup_temp_file(file_path: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Temporary file cleaned up", file_path=file_path)
    except Exception as e:
        logger.error(f"Failed to cleanup temporary file: {str(e)}", file_path=file_path)

@app.get("/celery-stats")
async def get_celery_stats():
    """Get Celery worker statistics"""
    try:
        inspector = celery_app.control.inspect()
        
        stats = inspector.stats()
        active_tasks = inspector.active()
        registered_tasks = inspector.registered()
        
        return {
            "stats": stats,
            "active_tasks": active_tasks,
            "registered_tasks": registered_tasks
        }
        
    except Exception as e:
        logger.error(f"Failed to get Celery stats: {str(e)}")
        return {"error": f"Failed to get Celery stats: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
