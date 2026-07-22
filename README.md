# Real-Time Sign Language Detection and Recognition Web Application Using Deep Learning

A comprehensive B.Tech capstone project demonstrating production-grade computer vision and deep learning implementation for American Sign Language (ASL) recognition via web interface.

## Project Overview

This application detects and recognizes sign language gestures in real-time from webcam feeds, uploaded images, and videos using deep learning models. The system converts predictions into readable text with optional text-to-speech synthesis, providing accessible communication tools for deaf and hard-of-hearing individuals.

**Status:** Under Development (Milestone 3/12)

## Technology Stack

### Frontend
- **Framework:** Next.js 15 + React 19
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **Animations:** Framer Motion
- **Webcam:** react-webcam

### Backend
- **Framework:** FastAPI (Python 3.12)
- **Vision:** OpenCV + MediaPipe Hands
- **Deep Learning:** PyTorch
- **Inference:** ONNX Runtime
- **Deployment:** Render

### Database & Services
- **Database:** PostgreSQL (Supabase)
- **Authentication:** Supabase Auth
- **File Storage:** AWS S3
- **Frontend Hosting:** Vercel
- **Monitoring:** Sentry + DataDog

## Quick Start

Detailed installation and setup instructions are provided in the documentation.

```bash
# Clone repository
git clone https://github.com/sxndy66/Sign-Language-Detection-and-Recognition-Web-Application-Using-Deep-Learning.git
cd Sign-Language-Detection-and-Recognition-Web-Application-Using-Deep-Learning

# Setup frontend
cd frontend
npm install
npm run dev

# Setup backend
cd ../backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload

# Setup ML pipeline
cd ../model
pip install -r requirements.txt
python src/train.py
```

## Project Structure

```
Sign-Language-Detection-and-Recognition-Web-Application-Using-Deep-Learning/
├── frontend/              # Next.js web application
├── backend/               # FastAPI server
├── model/                 # ML training & evaluation
├── docs/                  # Documentation & diagrams
├── deployment/            # Deployment configurations
├── scripts/               # Utility scripts
└── README.md
```

## Key Features

- ✅ Real-time webcam sign detection
- ✅ Image upload with batch processing
- ✅ Video upload with frame-by-frame analysis
- ✅ Prediction history with analytics
- ✅ Text-to-speech synthesis
- ✅ Dark mode support
- ✅ User authentication & profiles
- ✅ Admin dashboard
- ✅ Responsive design (mobile-first)
- ✅ WCAG 2.1 AA accessible

## Model Performance

**Target Metrics (ASL Alphabet - 26 classes):**
- Accuracy: 93-96%
- Precision: 92-95%
- Recall: 92-95%
- F1-Score: 92-95%
- Inference Latency: <200ms (CPU)

**Architecture:** ResNet50 + Transfer Learning

## Documentation

Comprehensive documentation includes:
- [Architecture Documentation](./docs/ARCHITECTURE.md)
- [API Documentation](./docs/API_DOCUMENTATION.md)
- [Database Schema](./docs/DATABASE_SCHEMA.md)
- [Deployment Guide](./docs/DEPLOYMENT_GUIDE.md)
- [User Manual](./docs/USER_MANUAL.md)
- [Installation Guide](./docs/INSTALLATION_GUIDE.md)

## Milestones

- [x] Milestone 1: Requirements gathering and problem framing
- [x] Milestone 2: System architecture and design
- [ ] Milestone 3: Folder structure and repository organization
- [ ] Milestone 4: Database design and SQL schema
- [ ] Milestone 5: Backend API development
- [ ] Milestone 6: Deep learning model design, training, and evaluation
- [ ] Milestone 7: API integration and inference pipeline
- [ ] Milestone 8: Frontend development and UI implementation
- [ ] Milestone 9: Testing and validation
- [ ] Milestone 10: Deployment preparation
- [ ] Milestone 11: Documentation generation
- [ ] Milestone 12: Final review and submission readiness

## License

MIT License - See LICENSE file for details

## Authors

**Development Team:** B.Tech Capstone Project
**Institution:** Computer Science Engineering
**Year:** 2024-2025

## Acknowledgements

- ASL Alphabet Dataset contributors
- MediaPipe by Google
- PyTorch community
- OpenCV contributors

---

**Last Updated:** January 2025
**Version:** 0.1.0-dev

For questions or support, please open an issue on GitHub.
