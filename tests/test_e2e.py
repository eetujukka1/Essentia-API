import pytest

from essentia_wrapper.audio import load_wav, normalize_audio
from essentia_wrapper.models.genres import GenrePredictor
from essentia_wrapper.models.instruments import InstrumentPredictor
from essentia_wrapper.models.vocal import VocalContentPredictor


class TestEndToEndGenre:
    """End-to-end tests for genre prediction pipeline."""

    @pytest.fixture
    def predictor(self):
        return GenrePredictor()

    def test_full_pipeline_single_file(self, predictor, sample_audio_path):
        """Test complete pipeline: load -> normalize -> predict genres."""
        sample_rate, audio_data = load_wav(str(sample_audio_path))
        normalized = normalize_audio(sample_rate, audio_data)
        result = predictor.predict(normalized)

        assert isinstance(result, list)
        assert len(result) == 5
        for item in result:
            assert "genre" in item
            assert "confidence" in item
            assert 0.0 <= item["confidence"] <= 1.0

    def test_full_pipeline_all_fixtures(self, predictor, audio_files):
        """Test pipeline works for all fixture audio files."""
        for audio_path in audio_files:
            sample_rate, audio_data = load_wav(str(audio_path))
            normalized = normalize_audio(sample_rate, audio_data)
            result = predictor.predict(normalized)

            assert isinstance(result, list)
            assert len(result) == 5


class TestEndToEndInstrument:
    """End-to-end tests for instrument prediction pipeline."""

    @pytest.fixture
    def predictor(self):
        return InstrumentPredictor()

    def test_full_pipeline_single_file(self, predictor, sample_audio_path):
        """Test complete pipeline: load -> normalize -> predict instruments."""
        sample_rate, audio_data = load_wav(str(sample_audio_path))
        normalized = normalize_audio(sample_rate, audio_data)
        result = predictor.predict(normalized)

        assert isinstance(result, list)
        assert len(result) == 5
        for item in result:
            assert "instrument" in item
            assert "confidence" in item
            assert 0.0 <= item["confidence"] <= 1.0

    def test_full_pipeline_all_fixtures(self, predictor, audio_files):
        """Test pipeline works for all fixture audio files."""
        for audio_path in audio_files:
            sample_rate, audio_data = load_wav(str(audio_path))
            normalized = normalize_audio(sample_rate, audio_data)
            result = predictor.predict(normalized)

            assert isinstance(result, list)
            assert len(result) == 5


class TestEndToEndVocal:
    """End-to-end tests for vocal content prediction pipeline."""

    @pytest.fixture
    def predictor(self):
        return VocalContentPredictor()

    def test_full_pipeline_predict(self, predictor, sample_audio_path):
        """Test complete pipeline: load -> normalize -> predict vocals."""
        sample_rate, audio_data = load_wav(str(sample_audio_path))
        normalized = normalize_audio(sample_rate, audio_data)
        result = predictor.predict(normalized)

        assert isinstance(result, dict)
        assert "has_vocals" in result
        assert "confidence" in result
        assert isinstance(result["has_vocals"], bool)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_full_pipeline_all_fixtures(self, predictor, audio_files):
        """Test pipeline works for all fixture audio files."""
        for audio_path in audio_files:
            sample_rate, audio_data = load_wav(str(audio_path))
            normalized = normalize_audio(sample_rate, audio_data)

            result = predictor.predict(normalized)

            assert isinstance(result, dict)
            assert isinstance(result["has_vocals"], bool)
            assert 0.0 <= result["confidence"] <= 1.0
