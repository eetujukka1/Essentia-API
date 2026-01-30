from flask import Request

from essentia_wrapper import (
    load_wav_bytes,
    normalize_audio,
    GenrePredictor,
    InstrumentPredictor,
    VocalContentPredictor,
)

genre_predictor = GenrePredictor()
instrument_predictor = InstrumentPredictor()
vocal_predictor = VocalContentPredictor()


def process_audio_request(request: Request):
    """Extract audio from a Flask request and return normalized audio data.

    Supports multipart file upload (field name 'file') or raw audio bytes in body.
    """
    if "file" in request.files:
        audio_bytes = request.files["file"].read()
    elif request.data:
        audio_bytes = request.data
    else:
        raise ValueError("No audio data provided. Send a file upload or raw audio bytes.")

    sample_rate, audio_data = load_wav_bytes(audio_bytes)
    return normalize_audio(sample_rate, audio_data)


def predict_genres(audio_data, top_n=5):
    return genre_predictor.predict(audio_data, top_n=top_n)


def predict_instruments(audio_data, top_n=5):
    return instrument_predictor.predict(audio_data, top_n=top_n)


def predict_vocals(audio_data):
    return vocal_predictor.predict(audio_data)
