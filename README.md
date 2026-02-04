# AI-Generated Voice Detection API

A FastAPI-based microservice that analyzes audio files to determine whether they contain AI-generated or human voices. The system provides confidence scores and language detection capabilities through a secure REST API.

## Features

- **AI Voice Detection**: Classify audio as AI-generated or human with confidence scores
- **Multiple Audio Formats**: Support for MP3, WAV, and FLAC formats
- **Secure Authentication**: API key-based authentication for all endpoints
- **Language Detection**: Detect or preserve language information from audio
- **Production Ready**: Designed for deployment on Render.com with health checks
- **Comprehensive Error Handling**: Clear error messages without exposing internal details
- **Property-Based Testing**: Extensive test coverage using Hypothesis for correctness guarantees

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Local Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-voice-detection-api
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create environment configuration:
```bash
cp .env.example .env
```

5. Edit `.env` file with your configuration (see [Configuration](#configuration))

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Required: Comma-separated list of valid API keys
API_KEYS=your-secret-key-1,your-secret-key-2

# Required: HuggingFace model name for audio classification
MODEL_NAME=facebook/wav2vec2-base

# Required: Directory for caching downloaded models
MODEL_CACHE_DIR=/tmp/models

# Optional: Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Optional: Server configuration (defaults shown)
HOST=0.0.0.0
PORT=8000
```

### Configuration Details

- **API_KEYS**: Generate secure random keys for production. Multiple keys can be provided for different clients.
- **MODEL_NAME**: Use a pretrained audio classification model from HuggingFace. Default is `facebook/wav2vec2-base`.
- **MODEL_CACHE_DIR**: Directory where models are cached. Use `/tmp/models` for Render.com or a persistent directory locally.
- **LOG_LEVEL**: Controls verbosity of logs. Use `INFO` for production, `DEBUG` for development.

## Usage

### Starting the Server

Run the API server locally:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, access the interactive API documentation:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Reference

### Authentication

All API endpoints (except `/health`) require authentication using an API key.

Include the API key in the request header:
```
x-api-key: your-secret-key
```

### Endpoints

#### POST /api/v1/detect

Analyze audio to detect if it contains AI-generated voices.

**Request Body:**

```json
{
  "language": "en",
  "audio_format": "mp3",
  "audio_base64": "BASE64_ENCODED_AUDIO_DATA"
}
```

**Request Fields:**
- `language` (string, required): Expected language code (e.g., "en", "es", "fr")
- `audio_format` (string, required): Audio format - "mp3", "wav", or "flac"
- `audio_base64` (string, required): Base64-encoded audio file data

**Response (200 OK):**

```json
{
  "is_ai_generated": true,
  "confidence": 0.87,
  "detected_language": "en",
  "message": "Voice detected as AI-generated with high confidence"
}
```

**Response Fields:**
- `is_ai_generated` (boolean): True if voice is AI-generated, false if human
- `confidence` (float): Confidence score between 0.0 and 1.0
- `detected_language` (string): Detected or provided language code
- `message` (string): Human-readable description of the result

**Example using curl:**

```bash
# First, encode your audio file to base64
AUDIO_BASE64=$(base64 -w 0 sample.mp3)

# Make the API request
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret-key" \
  -d "{
    \"language\": \"en\",
    \"audio_format\": \"mp3\",
    \"audio_base64\": \"$AUDIO_BASE64\"
  }"
```

**Example using Python:**

```python
import requests
import base64

# Read and encode audio file
with open("sample.mp3", "rb") as audio_file:
    audio_base64 = base64.b64encode(audio_file.read()).decode()

# Make API request
response = requests.post(
    "http://localhost:8000/api/v1/detect",
    headers={"x-api-key": "your-secret-key"},
    json={
        "language": "en",
        "audio_format": "mp3",
        "audio_base64": audio_base64
    }
)

result = response.json()
print(f"AI Generated: {result['is_ai_generated']}")
print(f"Confidence: {result['confidence']}")
```

#### GET /health

Health check endpoint for monitoring service availability.

**Response (200 OK):**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

**Example:**

```bash
curl http://localhost:8000/health
```

### Error Responses

The API returns structured error responses for various failure scenarios:

**401 Unauthorized** - Missing API key:
```json
{
  "error": "AuthenticationError",
  "detail": "API key required",
  "status_code": 401
}
```

**403 Forbidden** - Invalid API key:
```json
{
  "error": "AuthorizationError",
  "detail": "Invalid API key",
  "status_code": 403
}
```

**422 Unprocessable Entity** - Validation error:
```json
{
  "detail": [
    {
      "loc": ["body", "audio_base64"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**400 Bad Request** - Audio processing error:
```json
{
  "error": "AudioDecodingError",
  "detail": "Failed to decode audio data",
  "status_code": 400
}
```

**500 Internal Server Error** - Model inference error:
```json
{
  "error": "ModelInferenceError",
  "detail": "Prediction failed, please try again",
  "status_code": 500
}
```

## Deployment

### Deploying to Render.com

This API is configured for easy deployment on Render.com.

#### Prerequisites

- Render.com account
- GitHub repository with your code

#### Deployment Steps

1. **Push code to GitHub:**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Create new Web Service on Render:**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect the `render.yaml` configuration

3. **Configure Environment Variables:**
   - In the Render dashboard, go to your service settings
   - Add environment variables:
     - `API_KEYS`: Your production API keys (comma-separated)
     - `MODEL_NAME`: `facebook/wav2vec2-base` (or your chosen model)
     - `MODEL_CACHE_DIR`: `/tmp/models`
     - `LOG_LEVEL`: `INFO`

4. **Deploy:**
   - Click "Create Web Service"
   - Render will build and deploy your application
   - Monitor the deployment logs for any issues

5. **Verify Deployment:**
```bash
curl https://your-app-name.onrender.com/health
```

#### Render.yaml Configuration

The included `render.yaml` file configures:
- Python 3.10 environment
- Automatic dependency installation
- Health check monitoring at `/health`
- Environment variable management

### Docker Deployment (Optional)

Build and run using Docker:

```bash
# Build the image
docker build -t ai-voice-detection-api .

# Run the container
docker run -p 8000:8000 \
  -e API_KEYS=your-secret-key \
  -e MODEL_NAME=facebook/wav2vec2-base \
  -e MODEL_CACHE_DIR=/tmp/models \
  ai-voice-detection-api
```

## Testing

### Running Tests

Run the complete test suite:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_properties.py

# Run with verbose output
pytest -v
```

### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Property-Based Tests**: Test universal properties across many generated inputs using Hypothesis
- **Integration Tests**: Test complete API workflows end-to-end

### Property-Based Testing

This project uses Hypothesis for property-based testing to ensure correctness across a wide range of inputs:

```bash
# Run property tests with detailed output
pytest tests/test_properties.py -v

# Run with specific number of examples
pytest tests/test_properties.py --hypothesis-show-statistics
```

## Troubleshooting

### Common Issues

#### 1. Model Download Fails

**Problem**: Model fails to download from HuggingFace

**Solution**:
- Check internet connectivity
- Verify `MODEL_NAME` is correct
- Ensure `MODEL_CACHE_DIR` has write permissions
- Try downloading manually:
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("facebook/wav2vec2-base")
```

#### 2. Audio Decoding Errors

**Problem**: "Failed to decode audio data" error

**Solution**:
- Verify audio file is valid and not corrupted
- Check base64 encoding is correct:
```bash
base64 -d encoded.txt > decoded.mp3
# Try playing decoded.mp3
```
- Ensure audio format matches the `audio_format` field

#### 3. Authentication Failures

**Problem**: 401 or 403 errors

**Solution**:
- Verify API key is included in `x-api-key` header
- Check API key matches one in `API_KEYS` environment variable
- Ensure no extra whitespace in API key

#### 4. Memory Issues

**Problem**: Out of memory errors with large audio files

**Solution**:
- Limit audio file size (recommend max 10MB)
- Increase server memory allocation
- Process audio in chunks if possible

#### 5. Slow Predictions

**Problem**: API responses are slow

**Solution**:
- Model loads on first request (subsequent requests are faster)
- Consider using a smaller/faster model
- Enable model caching
- Use GPU acceleration if available

### Logging

Enable debug logging for troubleshooting:

```bash
# Set in .env file
LOG_LEVEL=DEBUG

# Or set environment variable
export LOG_LEVEL=DEBUG
uvicorn main:app --reload
```

Logs include:
- Request/response details
- Model loading status
- Feature extraction progress
- Error stack traces (server-side only)

### Health Check

Monitor service health:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

If `model_loaded` is `false`, the model failed to load. Check logs for details.

## Project Structure

```
ai-voice-detection-api/
├── api/
│   ├── __init__.py
│   ├── middleware.py          # Authentication middleware
│   ├── routes.py              # API endpoint definitions
│   └── schemas.py             # Pydantic request/response models
├── core/
│   ├── __init__.py
│   └── config.py              # Configuration management
├── services/
│   ├── __init__.py
│   ├── audio_processor.py     # Audio decoding and processing
│   ├── feature_extractor.py   # Audio feature extraction
│   └── model_service.py       # ML model loading and inference
├── utils/
│   ├── __init__.py
│   ├── exceptions.py          # Custom exception classes
│   └── logger.py              # Logging configuration
├── tests/
│   ├── __init__.py
│   ├── test_api_integration.py
│   ├── test_audio_processor.py
│   ├── test_authentication.py
│   ├── test_config.py
│   ├── test_feature_extractor.py
│   ├── test_health.py
│   ├── test_model_service.py
│   ├── test_properties.py     # Property-based tests
│   └── test_schemas.py
├── .env.example               # Example environment configuration
├── .gitignore
├── Dockerfile                 # Docker configuration
├── main.py                    # Application entry point
├── README.md                  # This file
├── render.yaml                # Render.com deployment config
└── requirements.txt           # Python dependencies
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Open an issue on GitHub
- Check the [Troubleshooting](#troubleshooting) section
- Review API documentation at `/docs` endpoint

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Audio processing with [librosa](https://librosa.org/)
- ML models from [HuggingFace](https://huggingface.co/)
- Property-based testing with [Hypothesis](https://hypothesis.readthedocs.io/)
