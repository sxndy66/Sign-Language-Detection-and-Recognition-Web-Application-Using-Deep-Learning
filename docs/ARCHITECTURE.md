# System Architecture Documentation

## Overview

The Sign Language Recognition system is built using a modern microservices-inspired architecture with clear separation of concerns:

- **Frontend:** Next.js + React (Presentation Layer)
- **Backend:** FastAPI (API Layer)
- **ML Pipeline:** PyTorch + ONNX (Intelligence Layer)
- **Database:** PostgreSQL (Data Layer)
- **Infrastructure:** Docker, Vercel, Render (Operations Layer)

## Architecture Diagram

```
┌────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Next.js + React Application             │  │
│  │  - Home | Detection | Dashboard | Admin         │  │
│  │  - Real-time webcam capture                     │  │
│  │  - File upload handling                         │  │
│  │  - Result visualization                         │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────┬─────────────────────────────────┘
                     │ HTTPS REST + WebSocket
┌────────────────────▼─────────────────────────────────┐
│                   API GATEWAY                         │
│  ┌──────────────────────────────────────────────────┐ │
│  │  FastAPI Application                             │ │
│  │  - Request validation & authentication           │ │
│  │  - Rate limiting & throttling                    │ │
│  │  - CORS & security headers                       │ │
│  └──────────────────────────────────────────────────┘ │
└────────────────┬──────────────────┬────────────────────┘
                 │                  │
    ┌────────────▼──────┐  ┌────────▼───────────┐
    │  Service Layer    │  │  ML Inference      │
    ├───────────────────┤  │  Service           │
    │ - Auth Service    │  ├───────────────────┤
    │ - User Service    │  │ - Model Loading   │
    │ - Upload Service  │  │ - Preprocessing   │
    │ - Storage Service │  │ - Inference       │
    │ - History Service │  │ - Post-processing │
    └────────┬──────────┘  └────────┬──────────┘
             │                      │
    ┌────────▼──────────────────────▼──────┐
    │      Data Access Layer               │
    ├──────────────────────────────────────┤
    │ SQLAlchemy ORM                       │
    │ Database connection pooling          │
    │ Query optimization                   │
    └────────┬──────────────────────────────┘
             │
    ┌────────▼──────────────────────────┐
    │    PostgreSQL Database             │
    ├───────────────────────���────────────┤
    │ - Users table                      │
    │ - Predictions table                │
    │ - Uploads table                    │
    │ - History & Audit tables           │
    │ - Model metadata                   │
    └────────────────────────────────────┘

    ┌────────────────────────────────────┐
    │    External Services               │
    ├────────────────────────────────────┤
    │ - AWS S3 (File Storage)            │
    │ - Redis (Caching)                  │
    │ - Supabase Auth                    │
    └────────────────────────────────────┘
```

## Component Descriptions

### Frontend Layer (Next.js + React)

**Responsibilities:**
- User interface rendering
- Webcam stream capture and display
- File upload handling
- Form validation
- State management
- Responsive design
- Accessibility compliance

**Key Features:**
- Server-side rendering (SSR) for performance
- Static site generation (SSG) for documentation pages
- Image optimization
- Font optimization
- Code splitting
- Dynamic imports

**Technology Stack:**
- Next.js 15 (App Router)
- React 19
- TypeScript
- Tailwind CSS
- Framer Motion
- React Webcam
- Axios (HTTP client)
- SWR (Data fetching)

### Backend API Layer (FastAPI)

**Responsibilities:**
- Request routing and validation
- Authentication & authorization
- Business logic implementation
- ML model inference orchestration
- Data persistence
- External service integration
- Error handling
- Logging and monitoring

**Key Features:**
- Async request handling
- Automatic API documentation (Swagger)
- Request validation (Pydantic)
- Dependency injection
- Middleware pipeline
- Background tasks
- Streaming responses

**Endpoints Structure:**
```
/api/v1/
├── /auth
│   ├── POST /register
│   ├── POST /login
│   ├── POST /logout
│   ├── POST /refresh
│   └── GET /me
├── /predict
│   ├── POST / (single prediction)
│   ├── GET /{id}
│   └── DELETE /{id}
├── /detect
│   ├── POST /webcam
│   └── POST /batch
├── /upload
│   ├── POST /image
│   ├── POST /video
│   ├── GET /
│   ├── GET /{id}
│   └── DELETE /{id}
├── /history
│   ├── GET /
│   ├── GET /stats
│   ├── GET /export
│   └── DELETE /clear
├── /users
│   ├── GET /profile
│   ├── PUT /profile
│   └── PUT /preferences
├── /admin
│   ├── GET /stats
│   ├── GET /users
│   ├── GET /logs
│   └── PUT /model-activate
└── /health
    ├── GET /
    └── GET /status
```

### ML Inference Service

**Responsibilities:**
- Load pre-trained models
- Image preprocessing
- Model inference
- Prediction post-processing
- Performance optimization

**ML Pipeline:**
```
Input Image (224×224×3)
    ↓
Preprocessing
  - Normalize pixel values
  - Apply augmentation (inference-time)
  - Hand detection (MediaPipe)
    ↓
Model Inference (ResNet50 + ONNX)
  - Forward pass through network
  - Output logits (26 classes)
    ↓
Post-Processing
  - Softmax to probabilities
  - Top-5 predictions extraction
  - Confidence filtering
    ↓
Output (Sign + Confidence)
```

**Model Details:**
- Architecture: ResNet50
- Input: 224×224×3 RGB image
- Output: 26 class probabilities (A-Z)
- Format: ONNX (CPU/GPU compatible)
- Size: ~100MB
- Inference Time: <200ms (CPU average)

### Database Layer

**Database Type:** PostgreSQL (Supabase)

**Key Tables:**

**users**
- User accounts, authentication, preferences
- Indexed on: email, username, role

**predictions**
- Individual sign predictions
- Stores: prediction, confidence, timestamp, user_id
- Indexed on: user_id, created_at, confidence

**uploads**
- File upload metadata and status
- Stores: filename, path, size, status, user_id
- Indexed on: user_id, status, created_at

**prediction_history**
- Aggregated user statistics
- Monthly rollups for analytics
- Indexed on: user_id, month_year

**audit_logs**
- System audit trail
- Stores: action, user, resource, result, timestamp
- Indexed on: user_id, action, created_at

**model_metadata**
- Model version tracking
- Performance metrics per version
- Indexed on: version, is_active

### Storage Layer

**File Storage:** AWS S3
- Upload images and videos
- CloudFront CDN for delivery
- Lifecycle policies (90-day retention)

**Cache Layer:** Redis
- Model predictions cache
- Session management
- Rate limiting counters
- TTL: 1 hour default

## Data Flow Diagrams

### Webcam Detection Flow

```
User Start Webcam
    ↓
Browser Capture Frame (30 FPS)
    ↓
MediaPipe Hand Detection (Client-side)
    ↓
Send Frame to Backend
    ↓
Backend Preprocessing
    ↓
ONNX Model Inference
    ↓
Post-Processing & Validation
    ↓
Store in Predictions Table
    ↓
WebSocket Response to Client
    ↓
Display on UI + Add to History
```

### Image Upload Flow

```
User Select Image
    ↓
Client Validation (type, size)
    ↓
Upload to S3 via Backend
    ↓
Store Upload Metadata in DB
    ↓
Backend Preprocessing
    ↓
ONNX Model Inference
    ↓
Store Prediction in DB
    ↓
Return Results to Client
    ↓
Display Results + Save to History
```

### Video Processing Flow

```
User Upload Video
    ↓
Store in S3
    ↓
Backend: Extract Frames (OpenCV)
    ↓
Process Each Frame (Batch)
    ↓
Run Inference on Each Frame
    ↓
Aggregate Results
    ↓
Create Timeline of Predictions
    ↓
Store in DB
    ↓
Return Timeline to Client
```

## Security Architecture

### Authentication Flow

```
User Credentials
    ↓
Request /auth/login
    ↓
Validate Email & Password (Bcrypt)
    ↓
Issue JWT Tokens
    ├── Access Token (15 min expiry)
    └── Refresh Token (7 days expiry)
    ↓
Store in httpOnly Cookies
    ↓
Client Authenticated
```

### Authorization Strategy

- **Role-Based Access Control (RBAC):** user, admin, moderator
- **Resource-Based:** Check user ownership on sensitive operations
- **Token Validation:** JWT verification on every request
- **Scope-based:** Permissions per endpoint

### Security Layers

1. **Transport Security:** TLS 1.3 (HTTPS)
2. **Request Validation:** Pydantic schemas, input sanitization
3. **Authentication:** JWT + httpOnly cookies
4. **Authorization:** Role-based middleware
5. **Rate Limiting:** Per-user, per-IP limits
6. **CORS:** Whitelist specific origins
7. **CSRF:** SameSite cookies, token validation
8. **Data Protection:** Encryption at rest, PII masking in logs

## Scalability Considerations

### Horizontal Scaling

- **Backend:** Multiple Render instances behind load balancer
- **Database:** Read replicas for query scaling
- **Cache:** Redis cluster for distributed caching
- **Storage:** S3 unlimited capacity with CDN

### Vertical Optimization

- **Database Indexing:** Strategic indexes on frequently queried columns
- **Query Optimization:** N+1 prevention, batch queries
- **Caching Strategy:** Cache-aside for hot data
- **Model Optimization:** ONNX quantization (INT8)

### Performance Targets

- API P95 latency: <500ms
- Frontend FCP: <2 seconds
- Database query P95: <100ms
- Model inference P95: <200ms (CPU), <50ms (GPU)

## Deployment Architecture

### Development Environment

```
Docker Compose
├── Frontend: Next.js dev server (port 3000)
├── Backend: FastAPI with reload (port 8000)
├── Database: PostgreSQL (port 5432)
└── Cache: Redis (port 6379)
```

### Staging Environment

```
GitHub Actions
    ↓
Build Docker images
    ↓
Deploy to staging servers
    ↓
Run smoke tests
    ↓
Ready for manual testing
```

### Production Environment

```
GitHub (main branch)
    ↓
CI/CD Pipeline
    ├── Lint & Test
    ├── Build
    ├── Security Scan
    └── Deploy
        ├── Frontend → Vercel
        ├── Backend → Render
        └── Database → Supabase
```

## Monitoring & Observability

### Metrics

- **API Metrics:** Request count, latency, error rate
- **Model Metrics:** Inference latency, throughput, error rate
- **Database Metrics:** Query latency, connection pool usage
- **System Metrics:** CPU, memory, disk, network

### Logging

- **Application Logs:** Structured JSON logging to Sentry
- **Access Logs:** Request/response logging (excluding sensitive data)
- **Audit Logs:** All user actions in database
- **Error Logs:** Stack traces and context

### Alerting

- Error rate >1%: Alert
- API latency P95 >1000ms: Alert
- Database connection pool >80%: Alert
- Disk usage >80%: Alert
- Deployment failure: Alert

---

**Last Updated:** January 2025  
**Version:** 1.0
