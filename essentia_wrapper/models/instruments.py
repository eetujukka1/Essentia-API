import json
import numpy as np
from essentia.standard import TensorflowPredictEffnetDiscogs, TensorflowPredict2D
from .base import BasePredictor, MODELS_DIR


class InstrumentPredictor(BasePredictor):
    """Predicts instruments present in audio."""

    def __init__(self):
        self._embedding_model = None
        self._classifier = None
        self._instrument_tags = None

    def _load_models(self):
        if self._embedding_model is not None:
            return

        metadata_path = MODELS_DIR / "mtg_jamendo_instrument-discogs-effnet-1.json"
        with open(metadata_path, "r") as f:
            model_data = json.load(f)
            self._instrument_tags = model_data["classes"]

        self._embedding_model = TensorflowPredictEffnetDiscogs(
            graphFilename=str(MODELS_DIR / "discogs-effnet-bs64-1.pb"),
            output="PartitionedCall:1",
        )

        self._classifier = TensorflowPredict2D(
            graphFilename=str(MODELS_DIR / "mtg_jamendo_instrument-discogs-effnet-1.pb")
        )

    def predict(self, audio_data: np.ndarray, top_n: int = 5) -> list[dict]:
        """
        Predict instruments present in audio.

        Args:
            audio_data: Normalized audio data (float32, mono, 16kHz).
            top_n: Number of top instruments to return.

        Returns:
            List of dicts with 'instrument' and 'confidence' keys,
            sorted by confidence descending.
        """
        self._load_models()

        embeddings = self._embedding_model(audio_data)
        predictions = self._classifier(embeddings)
        mean_predictions = np.mean(predictions, axis=0)

        instrument_predictions = list(zip(self._instrument_tags, mean_predictions))
        instrument_predictions.sort(key=lambda x: x[1], reverse=True)

        return [
            {"instrument": instrument, "confidence": round(float(score), 4)}
            for instrument, score in instrument_predictions[:top_n]
        ]
