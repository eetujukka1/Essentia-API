from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

MODELS_DIR = Path(__file__).parent.parent / "prediction_models"


class BasePredictor(ABC):
    """Base class for audio prediction models."""

    @abstractmethod
    def predict(self, audio_data: np.ndarray) -> any:
        """
        Run prediction on audio data.

        Args:
            audio_data: Normalized audio data (float32, mono, 16kHz).

        Returns:
            Prediction results (format depends on subclass).
        """
        pass
