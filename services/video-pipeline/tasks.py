from celery import Celery
import structlog
import cv2
import numpy as np
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Celery
celery_app = Celery(
    "video_pipeline",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
)

# Configure Celery settings
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
)

logger = structlog.get_logger()

@celery_app.task(bind=True)
def process_video_frames(self, video_path: str, frame_interval: int = 1) -> Dict[str, Any]:
    """
    Process video frames for face detection and analysis
    
    Args:
        video_path: Path to the video file
        frame_interval: Extract every Nth frame
        
    Returns:
        Dictionary with processing results
    """
    try:
        logger.info("Starting video frame processing", video_path=video_path, frame_interval=frame_interval)
        
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 0, 'status': 'Opening video file'}
        )
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info("Video properties", total_frames=total_frames, fps=fps, duration=duration)
        
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': total_frames, 'status': 'Processing frames'}
        )
        
        # Process frames
        processed_frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                # Process frame
                frame_result = process_single_frame(frame, frame_count)
                processed_frames.append(frame_result)
                
                # Update progress
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': frame_count,
                        'total': total_frames,
                        'status': f'Processed {len(processed_frames)} frames'
                    }
                )
            
            frame_count += 1
        
        cap.release()
        
        # Compile results
        results = {
            'video_path': video_path,
            'total_frames': total_frames,
            'processed_frames': len(processed_frames),
            'frame_interval': frame_interval,
            'fps': fps,
            'duration': duration,
            'frames': processed_frames
        }
        
        logger.info("Video processing completed", results=results)
        
        return results
        
    except Exception as e:
        logger.error("Video processing failed", error=str(e), video_path=video_path)
        raise
    
    finally:
        # Cleanup
        if 'cap' in locals():
            cap.release()

@celery_app.task(bind=True)
def extract_faces_from_video(self, video_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Extract faces from video frames
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted faces
        
    Returns:
        Dictionary with extraction results
    """
    try:
        logger.info("Starting face extraction from video", video_path=video_path)
        
        # Create output directory if specified
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Process video frames
        frame_results = process_video_frames.delay(video_path, frame_interval=5)
        frame_data = frame_results.get()
        
        # Extract faces from processed frames
        extracted_faces = []
        for frame_result in frame_data['frames']:
            if 'faces' in frame_result and frame_result['faces']:
                for face in frame_result['faces']:
                    face_data = {
                        'frame_number': frame_result['frame_number'],
                        'timestamp': frame_result['timestamp'],
                        'face_bbox': face['bbox'],
                        'face_quality': face['quality_metrics']
                    }
                    
                    # Save face image if output directory is specified
                    if output_dir:
                        face_filename = f"face_frame_{frame_result['frame_number']}_{face['face_id']}.jpg"
                        face_path = os.path.join(output_dir, face_filename)
                        cv2.imwrite(face_path, face['face_image'])
                        face_data['face_path'] = face_path
                    
                    extracted_faces.append(face_data)
        
        results = {
            'video_path': video_path,
            'total_frames_processed': len(frame_data['frames']),
            'faces_extracted': len(extracted_faces),
            'extracted_faces': extracted_faces
        }
        
        logger.info("Face extraction completed", results=results)
        return results
        
    except Exception as e:
        logger.error("Face extraction failed", error=str(e), video_path=video_path)
        raise

@celery_app.task(bind=True)
def analyze_video_liveness(self, video_path: str) -> Dict[str, Any]:
    """
    Analyze video for liveness detection
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with liveness analysis results
    """
    try:
        logger.info("Starting liveness analysis", video_path=video_path)
        
        # Process video frames
        frame_results = process_video_frames.delay(video_path, frame_interval=2)
        frame_data = frame_results.get()
        
        # Analyze liveness patterns
        liveness_analysis = analyze_liveness_patterns(frame_data['frames'])
        
        results = {
            'video_path': video_path,
            'liveness_score': liveness_analysis['liveness_score'],
            'is_live': liveness_analysis['is_live'],
            'confidence': liveness_analysis['confidence'],
            'analysis_details': liveness_analysis['details']
        }
        
        logger.info("Liveness analysis completed", results=results)
        return results
        
    except Exception as e:
        logger.error("Liveness analysis failed", error=str(e), video_path=video_path)
        raise

def process_single_frame(frame: np.ndarray, frame_number: int) -> Dict[str, Any]:
    """
    Process a single video frame
    
    Args:
        frame: Video frame as numpy array
        frame_number: Frame number in the video
        
    Returns:
        Dictionary with frame processing results
    """
    try:
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces in frame
        faces = detect_faces_in_frame(frame_rgb)
        
        # Calculate frame quality metrics
        quality_metrics = calculate_frame_quality(frame_rgb)
        
        frame_result = {
            'frame_number': frame_number,
            'timestamp': frame_number / 30.0,  # Assuming 30 FPS
            'frame_size': frame.shape,
            'quality_metrics': quality_metrics,
            'faces': faces
        }
        
        return frame_result
        
    except Exception as e:
        logger.error("Frame processing failed", error=str(e), frame_number=frame_number)
        return {
            'frame_number': frame_number,
            'error': str(e),
            'faces': []
        }

def detect_faces_in_frame(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect faces in a frame
    
    Args:
        frame: Frame as numpy array
        
    Returns:
        List of detected faces with metadata
    """
    try:
        # Load face detection model
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        face_results = []
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            face_region = frame[y:y+h, x:x+w]
            
            # Analyze face quality
            quality_metrics = analyze_face_quality(face_region)
            
            face_data = {
                'face_id': i,
                'bbox': (x, y, w, h),
                'face_image': face_region,
                'quality_metrics': quality_metrics
            }
            
            face_results.append(face_data)
        
        return face_results
        
    except Exception as e:
        logger.error("Face detection failed", error=str(e))
        return []

def calculate_frame_quality(frame: np.ndarray) -> Dict[str, Any]:
    """
    Calculate frame quality metrics
    
    Args:
        frame: Frame as numpy array
        
    Returns:
        Dictionary with quality metrics
    """
    try:
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Calculate metrics
        brightness = np.mean(gray)
        contrast = np.std(gray)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Quality score (0-100)
        quality_score = 0
        
        if 100 <= brightness <= 200:
            quality_score += 25
        elif 50 <= brightness <= 250:
            quality_score += 15
        
        if contrast > 50:
            quality_score += 25
        elif contrast > 30:
            quality_score += 15
        
        if sharpness > 100:
            quality_score += 25
        elif sharpness > 50:
            quality_score += 15
        
        if frame.shape[0] * frame.shape[1] >= 64000:  # 320x200 minimum
            quality_score += 25
        
        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'sharpness': float(sharpness),
            'quality_score': quality_score
        }
        
    except Exception as e:
        logger.error("Frame quality calculation failed", error=str(e))
        return {'quality_score': 0}

def analyze_face_quality(face_image: np.ndarray) -> Dict[str, Any]:
    """
    Analyze face image quality
    
    Args:
        face_image: Face image as numpy array
        
    Returns:
        Dictionary with face quality metrics
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        
        # Calculate metrics
        brightness = np.mean(gray)
        contrast = np.std(gray)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Quality score
        quality_score = 0
        
        if 100 <= brightness <= 200:
            quality_score += 25
        elif 50 <= brightness <= 250:
            quality_score += 15
        
        if contrast > 50:
            quality_score += 25
        elif contrast > 30:
            quality_score += 15
        
        if sharpness > 100:
            quality_score += 25
        elif sharpness > 50:
            quality_score += 15
        
        if face_image.shape[0] * face_image.shape[1] >= 2500:  # 50x50 minimum
            quality_score += 25
        
        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'sharpness': float(sharpness),
            'quality_score': quality_score
        }
        
    except Exception as e:
        logger.error("Face quality analysis failed", error=str(e))
        return {'quality_score': 0}

def analyze_liveness_patterns(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze frames for liveness patterns
    
    Args:
        frames: List of processed frames
        
    Returns:
        Dictionary with liveness analysis results
    """
    try:
        if not frames:
            return {
                'liveness_score': 0.0,
                'is_live': False,
                'confidence': 0.0,
                'details': 'No frames to analyze'
            }
        
        # Analyze face movement patterns
        face_movements = []
        face_sizes = []
        
        for frame in frames:
            if 'faces' in frame and frame['faces']:
                # Track face movements
                for face in frame['faces']:
                    bbox = face['bbox']
                    face_movements.append(bbox[:2])  # x, y coordinates
                    face_sizes.append(bbox[2] * bbox[3])  # width * height
        
        # Calculate liveness indicators
        liveness_score = 0.0
        confidence = 0.0
        details = []
        
        # Movement analysis
        if len(face_movements) > 1:
            movements = np.array(face_movements)
            movement_variance = np.var(movements, axis=0)
            total_movement = np.sum(np.linalg.norm(np.diff(movements, axis=0), axis=1))
            
            if total_movement > 10:  # Significant movement
                liveness_score += 0.3
                details.append("Detected natural face movement")
            else:
                details.append("Limited face movement detected")
        
        # Size variation analysis
        if len(face_sizes) > 1:
            size_variance = np.var(face_sizes)
            if size_variance > 1000:  # Size variation
                liveness_score += 0.3
                details.append("Detected face size variations")
            else:
                details.append("Consistent face size detected")
        
        # Quality consistency
        quality_scores = [frame.get('quality_metrics', {}).get('quality_score', 0) for frame in frames]
        avg_quality = np.mean(quality_scores)
        if avg_quality > 50:
            liveness_score += 0.2
            details.append("Good overall frame quality")
        
        # Face detection consistency
        frames_with_faces = sum(1 for frame in frames if 'faces' in frame and frame['faces'])
        face_consistency = frames_with_faces / len(frames)
        if face_consistency > 0.8:
            liveness_score += 0.2
            details.append("Consistent face detection")
        
        # Determine if live
        is_live = liveness_score > 0.5
        confidence = min(liveness_score, 1.0)
        
        return {
            'liveness_score': liveness_score,
            'is_live': is_live,
            'confidence': confidence,
            'details': details
        }
        
    except Exception as e:
        logger.error("Liveness pattern analysis failed", error=str(e))
        return {
            'liveness_score': 0.0,
            'is_live': False,
            'confidence': 0.0,
            'details': [f'Analysis failed: {str(e)}']
        }

if __name__ == "__main__":
    celery_app.start()
