import json
import numpy as np
from essentia.standard import TensorflowPredictMusiCNN, TensorflowPredict2D
from .base import BasePredictor, MODELS_DIR


class VocalContentPredictor(BasePredictor):
    """Predicts whether audio contains vocals or is instrumental."""

    def __init__(self):
        self._embedding_model = None
        self._classifier = None
        self._class_tags = None

    def _load_models(self):
        if self._embedding_model is not None:
            return

        metadata_path = MODELS_DIR / "voice_instrumental-msd-musicnn-1.json"
        with open(metadata_path, "r") as f:
            model_data = json.load(f)
            self._class_tags = model_data["classes"]

        self._embedding_model = TensorflowPredictMusiCNN(
            graphFilename=str(MODELS_DIR / "msd-musicnn-1.pb"),
            output="model/dense/BiasAdd",
        )

        self._classifier = TensorflowPredict2D(
            graphFilename=str(MODELS_DIR / "voice_instrumental-msd-musicnn-1.pb"),
            output="model/Softmax",
        )

    def predict(self, audio_data: np.ndarray) -> dict:
        """
        Predict whether audio contains vocals.

        Args:
            audio_data: Normalized audio data (float32, mono, 16kHz).

        Returns:
            Dict with 'has_vocals' bool and 'confidence' float.
        """
        self._load_models()

        embeddings = self._embedding_model(audio_data)
        predictions = self._classifier(embeddings)
        mean_predictions = predictions.mean(axis=0)

        class_predictions = list(zip(self._class_tags, mean_predictions))
        class_predictions.sort(key=lambda x: x[1], reverse=True)
        most_likely_class, confidence = class_predictions[0]

        return {
            "has_vocals": most_likely_class == "voice",
            "confidence": round(float(confidence), 4),
        }
