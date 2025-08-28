import hashlib
import hmac
import secrets
import base64
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from .config import get_config

class SecurityUtils:
    """Security utility functions"""
    
    def __init__(self):
        self.config = get_config()
        self.jwt_secret = self.config.security.jwt_secret
        self.jwt_algorithm = self.config.security.jwt_algorithm
        self.jwt_expire_minutes = self.config.security.jwt_expire_minutes
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate a cryptographically secure random token
        
        Args:
            length: Length of the token in bytes
            
        Returns:
            Base64 encoded secure token
        """
        return base64.urlsafe_b64encode(secrets.token_bytes(length)).decode('utf-8').rstrip('=')
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Dict[str, str]:
        """
        Hash a password using PBKDF2
        
        Args:
            password: Plain text password
            salt: Optional salt (generated if not provided)
            
        Returns:
            Dictionary with hashed password and salt
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 for password hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode('utf-8'),
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode('utf-8')))
        
        return {
            "hash": key.decode('utf-8'),
            "salt": salt
        }
    
    def verify_password(self, password: str, hash_value: str, salt: str) -> bool:
        """
        Verify a password against its hash
        
        Args:
            password: Plain text password to verify
            hash_value: Stored hash value
            salt: Salt used for hashing
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            # Generate hash with the same salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt.encode('utf-8'),
                iterations=100000,
            )
            
            key = base64.urlsafe_b64encode(kdf.derive(password.encode('utf-8')))
            computed_hash = key.decode('utf-8')
            
            # Use constant-time comparison to prevent timing attacks
            return hmac.compare_digest(computed_hash, hash_value)
        except Exception:
            return False
    
    def create_jwt_token(self, payload: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        Create a JWT token
        
        Args:
            payload: Token payload data
            expires_delta: Optional expiration time
            
        Returns:
            JWT token string
        """
        to_encode = payload.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.jwt_expire_minutes)
        
        to_encode.update({"exp": expire})
        
        return jwt.encode(to_encode, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.PyJWTError:
            return None
    
    def encrypt_data(self, data: str, key: Optional[str] = None) -> str:
        """
        Encrypt data using Fernet (symmetric encryption)
        
        Args:
            data: Data to encrypt
            key: Encryption key (generated if not provided)
            
        Returns:
            Encrypted data as base64 string
        """
        if key is None:
            key = Fernet.generate_key()
        
        f = Fernet(key)
        encrypted_data = f.encrypt(data.encode('utf-8'))
        
        # Return both encrypted data and key
        return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str, key: str) -> Optional[str]:
        """
        Decrypt data using Fernet
        
        Args:
            encrypted_data: Encrypted data as base64 string
            key: Decryption key
            
        Returns:
            Decrypted data if successful, None otherwise
        """
        try:
            f = Fernet(key)
            decoded_data = base64.urlsafe_b64decode(encrypted_data)
            decrypted_data = f.decrypt(decoded_data)
            return decrypted_data.decode('utf-8')
        except Exception:
            return None
    
    def generate_api_key(self, prefix: str = "club") -> str:
        """
        Generate a secure API key
        
        Args:
            prefix: Prefix for the API key
            
        Returns:
            Formatted API key
        """
        random_part = secrets.token_urlsafe(32)
        return f"{prefix}_{random_part}"
    
    def validate_api_key(self, api_key: str, prefix: str = "club") -> bool:
        """
        Validate API key format
        
        Args:
            api_key: API key to validate
            prefix: Expected prefix
            
        Returns:
            True if valid format, False otherwise
        """
        if not api_key or not api_key.startswith(f"{prefix}_"):
            return False
        
        # Check if the random part is at least 32 characters
        random_part = api_key[len(prefix) + 1:]
        return len(random_part) >= 32
    
    def create_hmac_signature(self, data: str, secret_key: str) -> str:
        """
        Create HMAC signature for data integrity
        
        Args:
            data: Data to sign
            secret_key: Secret key for signing
            
        Returns:
            HMAC signature
        """
        signature = hmac.new(
            secret_key.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(signature.digest()).decode('utf-8')
    
    def verify_hmac_signature(self, data: str, signature: str, secret_key: str) -> bool:
        """
        Verify HMAC signature
        
        Args:
            data: Original data
            signature: HMAC signature to verify
            secret_key: Secret key used for signing
            
        Returns:
            True if signature is valid, False otherwise
        """
        expected_signature = self.create_hmac_signature(data, secret_key)
        return hmac.compare_digest(expected_signature, signature)
    
    def sanitize_input(self, input_string: str) -> str:
        """
        Basic input sanitization
        
        Args:
            input_string: Input string to sanitize
            
        Returns:
            Sanitized string
        """
        if not input_string:
            return ""
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '{', '}']
        sanitized = input_string
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limit length
        max_length = 1000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    def validate_email(self, email: str) -> bool:
        """
        Basic email validation
        
        Args:
            email: Email address to validate
            
        Returns:
            True if valid email format, False otherwise
        """
        import re
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """
        Validate password strength
        
        Args:
            password: Password to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            "is_strong": True,
            "score": 0,
            "issues": [],
            "suggestions": []
        }
        
        # Length check
        if len(password) < 8:
            validation["is_strong"] = False
            validation["issues"].append("Password too short (minimum 8 characters)")
        elif len(password) >= 12:
            validation["score"] += 2
        else:
            validation["score"] += 1
        
        # Character variety checks
        if any(c.isupper() for c in password):
            validation["score"] += 1
        else:
            validation["issues"].append("No uppercase letters")
            validation["suggestions"].append("Add uppercase letters")
        
        if any(c.islower() for c in password):
            validation["score"] += 1
        else:
            validation["issues"].append("No lowercase letters")
            validation["suggestions"].append("Add lowercase letters")
        
        if any(c.isdigit() for c in password):
            validation["score"] += 1
        else:
            validation["issues"].append("No numbers")
            validation["suggestions"].append("Add numbers")
        
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            validation["score"] += 1
        else:
            validation["issues"].append("No special characters")
            validation["suggestions"].append("Add special characters")
        
        # Check for common patterns
        common_patterns = ["123", "abc", "qwerty", "password", "admin"]
        for pattern in common_patterns:
            if pattern.lower() in password.lower():
                validation["score"] -= 1
                validation["issues"].append(f"Contains common pattern: {pattern}")
                validation["suggestions"].append("Avoid common patterns")
        
        # Final score assessment
        if validation["score"] < 3:
            validation["is_strong"] = False
        
        return validation

# Global security utils instance
security_utils = SecurityUtils()
