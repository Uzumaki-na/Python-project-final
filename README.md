# Health Assessment and Disease Detection System

A comprehensive health assessment system that combines machine learning models for disease detection and health risk assessment.

## Features

- Health Risk Assessment based on personal and family medical history
- Skin Cancer Detection using deep learning
- Malaria Detection using deep learning
- Modern React TypeScript frontend
- FastAPI backend with ML model integration

## Tech Stack

- Frontend:
  - React 18+
  - TypeScript
  - Tailwind CSS
  - Vite

- Backend:
  - FastAPI
  - PyTorch
  - scikit-learn
  - Python 3.9+

## Setup

### Prerequisites

- Python 3.9+
- Node.js 16+

### Backend Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Create .env file:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the backend:
```bash
uvicorn main:app --reload
```

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Run the development server:
```bash
npm run dev
```

## API Documentation

Once the backend is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

Run backend tests:
```bash
cd backend
pytest
```

Run frontend tests:
```bash
cd frontend
npm test
```

## Production Deployment

1. Build frontend:
```bash
cd frontend
npm run build
```

2. Run backend with Gunicorn:
```bash
cd backend
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
