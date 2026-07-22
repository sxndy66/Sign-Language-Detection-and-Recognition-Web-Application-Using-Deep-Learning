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

## Quick Start

```bash
# Clone repository
git clone https://github.com/sxndy66/Sign-Language-Detection-and-Recognition-Web-Application-Using-Deep-Learning.git
cd Sign-Language-Detection-and-Recognition-Web-Application-Using-Deep-Learning

# Setup frontend
cd frontend
npm install
npm run dev

# Setup backend (in another terminal)
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

## Key Features

- ✅ Real-time webcam sign detection
- ✅ Image & video upload processing
- ✅ Prediction history with analytics
- ✅ Text-to-speech synthesis
- ✅ Dark mode support
- ✅ User authentication
- ✅ Admin dashboard
- ✅ Responsive mobile-first design
- ✅ WCAG 2.1 AA accessible

## Model Performance

**Target Metrics (ASL Alphabet - 26 classes):**
- Accuracy: 93-96%
- Architecture: ResNet50 + Transfer Learning

## Documentation

See [docs/](./docs/) folder for comprehensive documentation.

## License

MIT License - See LICENSE file

## Project Status

Currently implementing: Complete project structure

---

Version: 0.1.0-dev | Last Updated: January 2025