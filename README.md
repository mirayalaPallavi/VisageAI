<<<<<<< HEAD
# Club Project

A comprehensive microservices-based system for face detection, embedding generation, and similarity search, designed for club management and security applications.

## 🏗️ Architecture Overview

The Club Project is built using a microservices architecture with the following components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway  │    │   Auth Service  │    │ Template Service│
│   (Port 8000)  │    │   (Port 8001)   │    │   (Port 8002)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Image Pipeline  │    │ Video Pipeline  │    │Embedding Service│
│   (Port 8003)   │    │   (Port 8004)   │    │   (Port 8006)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Matching Service│
                    │   (Port 8007)   │
                    └─────────────────┘
```

## 🚀 Services

### 1. API Gateway (Port 8000)
- **Purpose**: Central entry point for all client requests
- **Features**: 
  - Request routing and load balancing
  - Authentication and authorization
  - Rate limiting and monitoring
  - Health checks for all services

### 2. Auth Service (Port 8001)
- **Purpose**: Handles user authentication and authorization
- **Features**:
  - JWT token generation and validation
  - User registration and login
  - Password hashing and security
  - OAuth2 integration

### 3. Template Service (Port 8002)
- **Purpose**: Manages club templates and metadata
- **Features**:
  - CRUD operations for templates
  - Version control for templates
  - Category and tag management
  - PostgreSQL database integration

### 4. Image Pipeline (Port 8003)
- **Purpose**: Processes single images for face detection
- **Features**:
  - Face detection using OpenCV
  - Face quality assessment
  - Image preprocessing
  - Integration with embedding service

### 5. Video Pipeline (Port 8004)
- **Purpose**: Processes video files for face analysis
- **Features**:
  - Frame extraction and processing
  - Asynchronous job processing with Celery
  - Liveness detection
  - Progress tracking

### 6. Embedding Service (Port 8006)
- **Purpose**: Generates face embeddings using deep learning
- **Features**:
  - PyTorch-based face embedding model
  - Batch processing capabilities
  - Model validation and quality checks
  - GPU acceleration support

### 7. Matching Service (Port 8007)
- **Purpose**: Performs vector similarity search
- **Features**:
  - FAISS integration for fast similarity search
  - Milvus vector database support
  - Multiple search algorithms
  - Scalable vector indexing

## 🛠️ Technology Stack

### Backend
- **Python 3.11**: Core programming language
- **FastAPI**: Modern, fast web framework
- **SQLAlchemy**: Database ORM
- **PostgreSQL**: Primary database
- **Redis**: Caching and message broker
- **Celery**: Asynchronous task processing

### AI/ML
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision library
- **FAISS**: Vector similarity search
- **Milvus**: Vector database
- **NumPy**: Numerical computing

### Infrastructure
- **Docker**: Containerization
- **Kubernetes**: Container orchestration
- **Helm**: Kubernetes package manager
- **GitHub Actions**: CI/CD pipeline

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Kubernetes cluster (for production)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/club-project.git
cd club-project
```

### 2. Environment Setup
Create a `.env` file in the root directory:
```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=club_project
POSTGRES_USER=club_user
POSTGRES_PASSWORD=club_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# JWT
JWT_SECRET=your-secret-key-here

# Environment
ENVIRONMENT=development
LOG_LEVEL=DEBUG
```

### 3. Start Services with Docker Compose
```bash
cd infra
docker-compose up -d
```

This will start all services:
- PostgreSQL database
- Redis cache
- All microservices
- Milvus vector database

### 4. Verify Services
Check if all services are running:
```bash
# API Gateway
curl http://localhost:8000/health

# Auth Service
curl http://localhost:8001/health

# Template Service
curl http://localhost:8002/health

# Image Pipeline
curl http://localhost:8003/health

# Video Pipeline
curl http://localhost:8004/health

# Embedding Service
curl http://localhost:8006/health

# Matching Service
curl http://localhost:8007/health
```

## 🔧 Development

### Running Individual Services
Each service can be run independently for development:

```bash
# API Gateway
cd services/api-gateway
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Auth Service
cd services/auth
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8001

# Template Service
cd services/template-service
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8002
```

### Database Migrations
```bash
cd services/template-service
alembic upgrade head
```

### Running Tests
```bash
# Run tests for all services
pytest services/*/tests/ -v

# Run tests for specific service
pytest services/api-gateway/tests/ -v
```

## 🚀 Production Deployment

### Kubernetes Deployment
```bash
# Apply namespace
kubectl apply -f infra/k8s/namespace.yaml

# Deploy all services
kubectl apply -f infra/k8s/

# Check deployment status
kubectl get pods -n club-project
```

### Helm Charts
```bash
# Install API Gateway
helm install api-gateway infra/helm-charts/api-gateway/

# Install Template Service
helm install template-service infra/helm-charts/template-service/
```

### Environment Variables for Production
```bash
# Production environment
ENVIRONMENT=production
LOG_LEVEL=INFO

# Secure JWT secret
JWT_SECRET=your-production-secret-key

# Database connection (use production credentials)
POSTGRES_HOST=your-production-db-host
POSTGRES_PASSWORD=your-production-db-password

# Redis connection
REDIS_HOST=your-production-redis-host
REDIS_PASSWORD=your-production-redis-password
```

## 📊 Monitoring and Observability

### Health Checks
All services provide health check endpoints:
- `/health`: Basic health status
- `/ready`: Readiness check with dependency verification
- `/metrics`: Prometheus metrics (where applicable)

### Logging
- Structured logging with `structlog`
- JSON format for production
- Configurable log levels
- Centralized log aggregation support

### Metrics
- Request counts and latencies
- Service-specific metrics
- Prometheus integration
- Grafana dashboards support

## 🔒 Security Features

- JWT-based authentication
- Password hashing with bcrypt
- Input sanitization
- CORS configuration
- Rate limiting
- Secure headers
- API key validation

## 📈 Performance

### Optimization Features
- Asynchronous processing with FastAPI
- Redis caching
- Database connection pooling
- Vector search optimization
- GPU acceleration for ML models

### Scaling
- Horizontal scaling with Kubernetes
- Load balancing
- Auto-scaling based on metrics
- Database read replicas
- Redis clustering

## 🧪 Testing

### Test Coverage
- Unit tests for all services
- Integration tests
- API endpoint testing
- Performance testing
- Security testing

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run tests with coverage
pytest --cov=services --cov-report=html

# Run specific test categories
pytest -m "unit"
pytest -m "integration"
pytest -m "api"
```

## 📚 API Documentation

### Interactive API Docs
- **Swagger UI**: Available at `/docs` for each service
- **ReDoc**: Available at `/redoc` for each service
- **OpenAPI Specification**: Available at `/openapi.json`

### API Examples
```bash
# Get API documentation
curl http://localhost:8000/docs

# Authenticate user
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=johndoe&password=secret"

# Process image
curl -X POST "http://localhost:8000/images/process-image" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "image=@path/to/image.jpg"
```

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run linting and tests
6. Submit a pull request

### Code Standards
- Follow PEP 8 for Python code
- Use type hints
- Write comprehensive docstrings
- Maintain test coverage above 80%
- Use conventional commit messages

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help
- **Documentation**: Check this README and service-specific docs
- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact the team at team@club-project.com

### Common Issues
- **Service won't start**: Check environment variables and dependencies
- **Database connection failed**: Verify PostgreSQL is running and credentials are correct
- **Redis connection failed**: Ensure Redis is running and accessible
- **Model loading failed**: Check if ML models are properly downloaded

## 🔮 Roadmap

### Upcoming Features
- [ ] Real-time face recognition
- [ ] Advanced liveness detection
- [ ] Multi-modal authentication
- [ ] Advanced analytics dashboard
- [ ] Mobile SDK
- [ ] Cloud deployment automation

### Version History
- **v1.0.0**: Initial release with core functionality
- **v1.1.0**: Enhanced security features
- **v1.2.0**: Performance optimizations
- **v2.0.0**: Advanced ML capabilities

---

**Built with ❤️ towards theProject **
=======
# VisageAI
>>>>>>> 951b467e1a95fa77b7f02656c2318aed21665177
