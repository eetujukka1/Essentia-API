# Essentia-API

A Flask REST API that exposes audio ML prediction models for genre classification, instrument detection, and vocal content detection via HTTP endpoints. Built on top of an [Essentia wrapper](https://github.com/eetujukka1/Essentia-Wrapper), a library for audio and music analysis.

## Features

- **Genre Classification** -- Predicts music genres using a Discogs-trained EffNet embedding model. Returns top-N genres with confidence scores.
- **Instrument Detection** -- Identifies instruments present in audio using the same EffNet embeddings with a Jamendo instrument classifier. Returns top-N instruments with confidence scores.
- **Vocal Content Detection** -- Determines whether audio contains vocals or is instrumental, using a MusiCNN embedding model. Returns a boolean result with a confidence score.

All models use lazy loading (loaded on first request, cached afterward) and expect WAV audio input.

## Architecture

The project follows a two-layer design:

1. **`essentia_wrapper/`** -- Self-contained audio processing and ML prediction library. Handles audio loading, normalization, and model inference.
2. **`app/`** -- Flask API layer that wraps `essentia_wrapper` to serve predictions over HTTP. Uses the app factory pattern with versioned endpoints under `/api/v1/`.

### Audio Processing Pipeline

```
WAV file or bytes
  -> load_wav() / load_wav_bytes()
  -> normalize_audio()  (mono, float32, 16 kHz)
  -> Predictor.predict()
  -> JSON response
```

## Project Structure

```
Essentia-API/
├── app/                        # Flask API layer
│   ├── __init__.py             # App factory
│   ├── config.py               # Configuration classes
│   ├── api/
│   │   └── v1/
│   │       ├── routes.py       # Route definitions
│   │       ├── schemas.py      # Request/response schemas
│   │       └── services.py     # Business logic
│   ├── errors/
│   │   └── handlers.py        # Error handlers
│   └── utils/
│       └── helpers.py         # Utility functions
├── essentia_wrapper/           # Audio ML prediction library
│   ├── audio.py                # Audio loading and normalization
│   └── models/
│       ├── base.py             # BasePredictor abstract class
│       ├── genres.py           # GenrePredictor
│       ├── instruments.py      # InstrumentPredictor
│       └── vocal.py            # VocalContentPredictor
├── tests/                      # Test suite
│   ├── conftest.py             # Shared pytest fixtures
│   └── fixtures/               # Sample WAV files
├── .env                        # Environment config (PORT)
├── requirements.txt
├── run.py                      # Dev server entrypoint
└── wsgi.py                     # Production entrypoint (gunicorn)
```

## Prerequisites

- Python 3.x
- WSL (on Windows)

## Setup

```bash
# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Development server

```bash
source venv/bin/activate
python run.py
```

The server reads its port from the `PORT` variable in `.env` (default: 3000).

### Production

```bash
source venv/bin/activate
gunicorn wsgi:app
```

### Running tests

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

## API Endpoints

All endpoints are versioned under `/api/v1/` and accept POST requests with WAV file uploads.

| Endpoint | Description | Response |
|---|---|---|
| `POST /api/v1/predict/genre` | Classify music genre | `[{"genre": str, "confidence": float}, ...]` |
| `POST /api/v1/predict/instrument` | Detect instruments | `[{"instrument": str, "confidence": float}, ...]` |
| `POST /api/v1/predict/vocal` | Detect vocal content | `{"has_vocals": bool, "confidence": float}` |

## Dependencies

- **Flask** -- Web framework
- **essentia-tensorflow** -- Audio ML models and Essentia framework
- **scipy** -- Audio I/O and signal processing
- **numpy** -- Numerical computations
- **pytest** -- Testing framework
