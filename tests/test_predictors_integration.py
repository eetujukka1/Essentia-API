import pytest
import numpy as np

from essentia_wrapper.models.genres import GenrePredictor
from essentia_wrapper.models.instruments import InstrumentPredictor
from essentia_wrapper.models.vocal import VocalContentPredictor


class TestGenrePredictorIntegration:
    """Integration tests for GenrePredictor."""

    @pytest.fixture
    def predictor(self):
        return GenrePredictor()

    def test_output_structure(self, predictor, sample_audio_normalized):
        """Test that predict returns correctly structured output."""
        result = predictor.predict(sample_audio_normalized)

        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict)
            assert "genre" in item
            assert "confidence" in item
            assert isinstance(item["genre"], str)
            assert isinstance(item["confidence"], float)
            assert 0.0 <= item["confidence"] <= 1.0

    def test_top_n_default(self, predictor, sample_audio_normalized):
        """Test that default top_n returns 5 results."""
        result = predictor.predict(sample_audio_normalized)
        assert len(result) == 5

    @pytest.mark.parametrize("top_n", [1, 3, 5, 10])
    def test_top_n_respected(self, predictor, sample_audio_normalized, top_n):
        """Test that top_n parameter limits results correctly."""
        result = predictor.predict(sample_audio_normalized, top_n=top_n)
        assert len(result) == top_n

    def test_results_sorted_by_confidence(self, predictor, sample_audio_normalized):
        """Test that results are sorted by confidence descending."""
        result = predictor.predict(sample_audio_normalized, top_n=10)
        confidences = [item["confidence"] for item in result]
        assert confidences == sorted(confidences, reverse=True)


class TestInstrumentPredictorIntegration:
    """Integration tests for InstrumentPredictor."""

    @pytest.fixture
    def predictor(self):
        return InstrumentPredictor()

    def test_output_structure(self, predictor, sample_audio_normalized):
        """Test that predict returns correctly structured output."""
        result = predictor.predict(sample_audio_normalized)

        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict)
            assert "instrument" in item
            assert "confidence" in item
            assert isinstance(item["instrument"], str)
            assert isinstance(item["confidence"], float)
            assert 0.0 <= item["confidence"] <= 1.0

    def test_top_n_default(self, predictor, sample_audio_normalized):
        """Test that default top_n returns 5 results."""
        result = predictor.predict(sample_audio_normalized)
        assert len(result) == 5

    @pytest.mark.parametrize("top_n", [1, 3, 5, 10])
    def test_top_n_respected(self, predictor, sample_audio_normalized, top_n):
        """Test that top_n parameter limits results correctly."""
        result = predictor.predict(sample_audio_normalized, top_n=top_n)
        assert len(result) == top_n

    def test_results_sorted_by_confidence(self, predictor, sample_audio_normalized):
        """Test that results are sorted by confidence descending."""
        result = predictor.predict(sample_audio_normalized, top_n=10)
        confidences = [item["confidence"] for item in result]
        assert confidences == sorted(confidences, reverse=True)


class TestVocalContentPredictorIntegration:
    """Integration tests for VocalContentPredictor."""

    @pytest.fixture
    def predictor(self):
        return VocalContentPredictor()

    def test_output_structure(self, predictor, sample_audio_normalized):
        """Test that predict returns correctly structured output."""
        result = predictor.predict(sample_audio_normalized)

        assert isinstance(result, dict)
        assert "has_vocals" in result
        assert "confidence" in result
        assert isinstance(result["has_vocals"], bool)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0


class TestGenrePredictorLazyLoading:
    """Tests for GenrePredictor lazy loading behavior."""

    def test_models_not_loaded_on_init(self):
        """Test that models are not loaded during __init__."""
        predictor = GenrePredictor()
        assert predictor._embedding_model is None
        assert predictor._classifier is None
        assert predictor._genre_tags is None

    def test_models_loaded_on_first_predict(self, sample_audio_normalized):
        """Test that models are loaded on first predict() call."""
        predictor = GenrePredictor()
        assert predictor._embedding_model is None

        predictor.predict(sample_audio_normalized)

        assert predictor._embedding_model is not None
        assert predictor._classifier is not None
        assert predictor._genre_tags is not None

    def test_models_cached_across_predictions(self, sample_audio_normalized):
        """Test that models are cached and not reloaded on subsequent calls."""
        predictor = GenrePredictor()
        predictor.predict(sample_audio_normalized)

        embedding_model_id = id(predictor._embedding_model)
        classifier_id = id(predictor._classifier)
        tags_id = id(predictor._genre_tags)

        predictor.predict(sample_audio_normalized)

        assert id(predictor._embedding_model) == embedding_model_id
        assert id(predictor._classifier) == classifier_id
        assert id(predictor._genre_tags) == tags_id


class TestInstrumentPredictorLazyLoading:
    """Tests for InstrumentPredictor lazy loading behavior."""

    def test_models_not_loaded_on_init(self):
        """Test that models are not loaded during __init__."""
        predictor = InstrumentPredictor()
        assert predictor._embedding_model is None
        assert predictor._classifier is None
        assert predictor._instrument_tags is None

    def test_models_loaded_on_first_predict(self, sample_audio_normalized):
        """Test that models are loaded on first predict() call."""
        predictor = InstrumentPredictor()
        assert predictor._embedding_model is None

        predictor.predict(sample_audio_normalized)

        assert predictor._embedding_model is not None
        assert predictor._classifier is not None
        assert predictor._instrument_tags is not None

    def test_models_cached_across_predictions(self, sample_audio_normalized):
        """Test that models are cached and not reloaded on subsequent calls."""
        predictor = InstrumentPredictor()
        predictor.predict(sample_audio_normalized)

        embedding_model_id = id(predictor._embedding_model)
        classifier_id = id(predictor._classifier)
        tags_id = id(predictor._instrument_tags)

        predictor.predict(sample_audio_normalized)

        assert id(predictor._embedding_model) == embedding_model_id
        assert id(predictor._classifier) == classifier_id
        assert id(predictor._instrument_tags) == tags_id


class TestVocalPredictorLazyLoading:
    """Tests for VocalContentPredictor lazy loading behavior."""

    def test_models_not_loaded_on_init(self):
        """Test that models are not loaded during __init__."""
        predictor = VocalContentPredictor()
        assert predictor._embedding_model is None
        assert predictor._classifier is None
        assert predictor._class_tags is None

    def test_models_loaded_on_first_predict(self, sample_audio_normalized):
        """Test that models are loaded on first predict() call."""
        predictor = VocalContentPredictor()
        assert predictor._embedding_model is None

        predictor.predict(sample_audio_normalized)

        assert predictor._embedding_model is not None
        assert predictor._classifier is not None
        assert predictor._class_tags is not None

    def test_models_cached_across_predictions(self, sample_audio_normalized):
        """Test that models are cached and not reloaded on subsequent calls."""
        predictor = VocalContentPredictor()
        predictor.predict(sample_audio_normalized)

        embedding_model_id = id(predictor._embedding_model)
        classifier_id = id(predictor._classifier)
        tags_id = id(predictor._class_tags)

        predictor.predict(sample_audio_normalized)

        assert id(predictor._embedding_model) == embedding_model_id
        assert id(predictor._classifier) == classifier_id
        assert id(predictor._class_tags) == tags_id


class TestGenrePredictorEdgeCases:
    """Edge case tests for GenrePredictor."""

    @pytest.fixture
    def predictor(self):
        return GenrePredictor()

    def test_empty_audio_array(self, predictor):
        """Test that empty audio array raises an error."""
        empty_audio = np.array([], dtype=np.float32)
        with pytest.raises(Exception):
            predictor.predict(empty_audio)

    def test_invalid_audio_shape_3d(self, predictor):
        """Test that 3D audio array raises an error."""
        audio_3d = np.zeros((10, 10, 10), dtype=np.float32)
        with pytest.raises(Exception):
            predictor.predict(audio_3d)

    def test_invalid_audio_shape_2d(self, predictor):
        """Test that 2D audio array raises an error."""
        audio_2d = np.zeros((100, 2), dtype=np.float32)
        with pytest.raises(Exception):
            predictor.predict(audio_2d)

    def test_wrong_dtype(self, predictor):
        """Test behavior with wrong dtype (int instead of float32)."""
        audio_int = np.zeros(16000, dtype=np.int16)
        with pytest.raises(Exception):
            predictor.predict(audio_int)

    def test_none_input(self, predictor):
        """Test that None input raises an error."""
        with pytest.raises(Exception):
            predictor.predict(None)

    def test_non_array_input(self, predictor):
        """Test that non-array input raises an error."""
        with pytest.raises(Exception):
            predictor.predict([0.1, 0.2, 0.3])


class TestInstrumentPredictorEdgeCases:
    """Edge case tests for InstrumentPredictor."""

    @pytest.fixture
    def predictor(self):
        return InstrumentPredictor()

    def test_empty_audio_array(self, predictor):
        """Test that empty audio array raises an error."""
        empty_audio = np.array([], dtype=np.float32)
        with pytest.raises(Exception):
            predictor.predict(empty_audio)

    def test_invalid_audio_shape_3d(self, predictor):
        """Test that 3D audio array raises an error."""
        audio_3d = np.zeros((10, 10, 10), dtype=np.float32)
        with pytest.raises(Exception):
            predictor.predict(audio_3d)

    def test_invalid_audio_shape_2d(self, predictor):
        """Test that 2D audio array raises an error."""
        audio_2d = np.zeros((100, 2), dtype=np.float32)
        with pytest.raises(Exception):
            predictor.predict(audio_2d)

    def test_wrong_dtype(self, predictor):
        """Test behavior with wrong dtype (int instead of float32)."""
        audio_int = np.zeros(16000, dtype=np.int16)
        with pytest.raises(Exception):
            predictor.predict(audio_int)

    def test_none_input(self, predictor):
        """Test that None input raises an error."""
        with pytest.raises(Exception):
            predictor.predict(None)

    def test_non_array_input(self, predictor):
        """Test that non-array input raises an error."""
        with pytest.raises(Exception):
            predictor.predict([0.1, 0.2, 0.3])


class TestVocalPredictorEdgeCases:
    """Edge case tests for VocalContentPredictor."""

    @pytest.fixture
    def predictor(self):
        return VocalContentPredictor()

    def test_empty_audio_array(self, predictor):
        """Test that empty audio array raises an error."""
        empty_audio = np.array([], dtype=np.float32)
        with pytest.raises(Exception):
            predictor.predict(empty_audio)

    def test_invalid_audio_shape_3d(self, predictor):
        """Test that 3D audio array raises an error."""
        audio_3d = np.zeros((10, 10, 10), dtype=np.float32)
        with pytest.raises(Exception):
            predictor.predict(audio_3d)

    def test_invalid_audio_shape_2d(self, predictor):
        """Test that 2D audio array raises an error."""
        audio_2d = np.zeros((100, 2), dtype=np.float32)
        with pytest.raises(Exception):
            predictor.predict(audio_2d)

    def test_wrong_dtype(self, predictor):
        """Test behavior with wrong dtype (int instead of float32)."""
        audio_int = np.zeros(16000, dtype=np.int16)
        with pytest.raises(Exception):
            predictor.predict(audio_int)

    def test_none_input(self, predictor):
        """Test that None input raises an error."""
        with pytest.raises(Exception):
            predictor.predict(None)

    def test_non_array_input(self, predictor):
        """Test that non-array input raises an error."""
        with pytest.raises(Exception):
            predictor.predict([0.1, 0.2, 0.3])
