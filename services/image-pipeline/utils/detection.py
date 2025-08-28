import cv2
import numpy as np
from typing import List, Tuple, Optional
import structlog

logger = structlog.get_logger()

class FaceDetector:
    """Face detection utility using OpenCV"""
    
    def __init__(self):
        # Load pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Alternative: use DNN-based face detector for better accuracy
        try:
            self.face_detector = cv2.dnn.readNet(
                "models/opencv_face_detector_uint8.pb",
                "models/opencv_face_detector.pbtxt"
            )
            self.use_dnn = True
            logger.info("DNN face detector loaded successfully")
        except:
            self.use_dnn = False
            logger.info("Using Haar cascade face detector")
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of bounding boxes (x, y, width, height)
        """
        if self.use_dnn:
            return self._detect_faces_dnn(image)
        else:
            return self._detect_faces_haar(image)
    
    def _detect_faces_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces.tolist()
    
    def _detect_faces_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using DNN"""
        height, width = image.shape[:2]
        
        # Prepare input blob
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (300, 300), [104, 117, 123], False, False
        )
        
        # Set input and get detections
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:  # Confidence threshold
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                
                faces.append((x1, y1, x2 - x1, y2 - y1))
        
        return faces
    
    def extract_face_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract face region from image
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            Cropped face image
        """
        x, y, w, h = bbox
        
        # Ensure coordinates are within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        # Extract face region
        face_region = image[y:y+h, x:x+w]
        
        return face_region
    
    def preprocess_face(self, face_image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess face image for model input
        
        Args:
            face_image: Input face image
            target_size: Target size for the model
            
        Returns:
            Preprocessed face image
        """
        # Resize to target size
        resized = cv2.resize(face_image, target_size)
        
        # Convert to RGB if needed
        if len(resized.shape) == 3:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            rgb = resized
        
        # Normalize pixel values
        normalized = rgb.astype(np.float32) / 255.0
        
        return normalized
    
    def detect_and_extract_faces(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
        """
        Detect faces and extract preprocessed face regions
        
        Args:
            image: Input image
            target_size: Target size for extracted faces
            
        Returns:
            List of preprocessed face images
        """
        # Detect faces
        faces = self.detect_faces(image)
        
        extracted_faces = []
        for bbox in faces:
            # Extract face region
            face_region = self.extract_face_region(image, bbox)
            
            # Preprocess face
            preprocessed = self.preprocess_face(face_region, target_size)
            
            extracted_faces.append(preprocessed)
        
        logger.info(f"Extracted {len(extracted_faces)} faces from image")
        return extracted_faces
    
    def validate_face_quality(self, face_image: np.ndarray) -> dict:
        """
        Validate face image quality
        
        Args:
            face_image: Input face image
            
        Returns:
            Dictionary with quality metrics
        """
        # Convert to grayscale for analysis
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_image
        
        # Calculate quality metrics
        quality_metrics = {
            'brightness': np.mean(gray),
            'contrast': np.std(gray),
            'sharpness': self._calculate_sharpness(gray),
            'face_size': face_image.shape[0] * face_image.shape[1],
            'aspect_ratio': face_image.shape[1] / face_image.shape[0]
        }
        
        # Quality score (0-100)
        quality_score = 0
        
        # Brightness check (ideal: 100-200)
        if 100 <= quality_metrics['brightness'] <= 200:
            quality_score += 25
        elif 50 <= quality_metrics['brightness'] <= 250:
            quality_score += 15
        
        # Contrast check (higher is better)
        if quality_metrics['contrast'] > 50:
            quality_score += 25
        elif quality_metrics['contrast'] > 30:
            quality_score += 15
        
        # Sharpness check
        if quality_metrics['sharpness'] > 100:
            quality_score += 25
        elif quality_metrics['sharpness'] > 50:
            quality_score += 15
        
        # Size check (minimum 50x50 pixels)
        if quality_metrics['face_size'] >= 2500:
            quality_score += 25
        elif quality_metrics['face_size'] >= 1000:
            quality_score += 15
        
        quality_metrics['quality_score'] = quality_score
        
        return quality_metrics
    
    def _calculate_sharpness(self, gray_image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        return laplacian.var()
