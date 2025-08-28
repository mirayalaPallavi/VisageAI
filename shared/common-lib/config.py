import os
from typing import Optional, Any, Dict
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseConfig(BaseSettings):
    """Database configuration settings"""
    
    host: str = Field(default="localhost", env="POSTGRES_HOST")
    port: int = Field(default=5432, env="POSTGRES_PORT")
    database: str = Field(default="club_project", env="POSTGRES_DB")
    username: str = Field(default="club_user", env="POSTGRES_USER")
    password: str = Field(default="", env="POSTGRES_PASSWORD")
    
    @property
    def url(self) -> str:
        """Get database connection URL"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    class Config:
        env_file = ".env"

class RedisConfig(BaseSettings):
    """Redis configuration settings"""
    
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    database: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    @property
    def url(self) -> str:
        """Get Redis connection URL"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.database}"
        return f"redis://{self.host}:{self.port}/{self.database}"
    
    class Config:
        env_file = ".env"

class SecurityConfig(BaseSettings):
    """Security configuration settings"""
    
    jwt_secret: str = Field(default="your-secret-key-here", env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=30, env="JWT_EXPIRE_MINUTES")
    
    # CORS settings
    allowed_origins: list = Field(default=["*"], env="ALLOWED_ORIGINS")
    allowed_methods: list = Field(default=["GET", "POST", "PUT", "DELETE"], env="ALLOWED_METHODS")
    allowed_headers: list = Field(default=["*"], env="ALLOWED_HEADERS")
    
    class Config:
        env_file = ".env"

class LoggingConfig(BaseSettings):
    """Logging configuration settings"""
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="json", env="LOG_FORMAT")
    output_file: Optional[str] = Field(default=None, env="LOG_OUTPUT_FILE")
    
    class Config:
        env_file = ".env"

class AppConfig(BaseSettings):
    """Main application configuration"""
    
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Service URLs
    api_gateway_url: str = Field(default="http://localhost:8000", env="API_GATEWAY_URL")
    auth_service_url: str = Field(default="http://localhost:8001", env="AUTH_SERVICE_URL")
    template_service_url: str = Field(default="http://localhost:8002", env="TEMPLATE_SERVICE_URL")
    image_pipeline_url: str = Field(default="http://localhost:8003", env="IMAGE_PIPELINE_URL")
    video_pipeline_url: str = Field(default="http://localhost:8004", env="VIDEO_PIPELINE_URL")
    embedding_service_url: str = Field(default="http://localhost:8006", env="EMBEDDING_SERVICE_URL")
    matching_service_url: str = Field(default="http://localhost:8007", env="MATCHING_SERVICE_URL")
    
    # Database and Redis
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    
    # Security
    security: SecurityConfig = SecurityConfig()
    
    # Logging
    logging: LoggingConfig = LoggingConfig()
    
    class Config:
        env_file = ".env"

def get_config() -> AppConfig:
    """Get application configuration"""
    return AppConfig()

def get_env_var(key: str, default: Any = None) -> Any:
    """Get environment variable with fallback"""
    return os.getenv(key, default)

def get_env_var_required(key: str) -> str:
    """Get required environment variable"""
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    return value

def is_production() -> bool:
    """Check if running in production environment"""
    return get_env_var("ENVIRONMENT", "development").lower() == "production"

def is_development() -> bool:
    """Check if running in development environment"""
    return get_env_var("ENVIRONMENT", "development").lower() == "development"

def is_testing() -> bool:
    """Check if running in testing environment"""
    return get_env_var("ENVIRONMENT", "development").lower() == "testing"

# Global configuration instance
config = get_config()
