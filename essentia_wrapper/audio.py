from scipy.io import wavfile
from scipy import signal
import numpy as np
import io


def load_wav(file_path: str) -> tuple[int, np.ndarray]:
    """Load a WAV file from disk."""
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data


def load_wav_bytes(wav_bytes: bytes) -> tuple[int, np.ndarray]:
    """Load a WAV file from bytes."""
    sample_rate, data = wavfile.read(io.BytesIO(wav_bytes))
    return sample_rate, data


def normalize_audio(sample_rate: int, audio_data: np.ndarray, target_sr: int = 16000) -> np.ndarray:
    """
    Normalize and resample audio data.

    Converts audio data to float32 format normalized to [-1, 1].
    Converts stereo to mono and resamples to target sample rate.

    Args:
        sample_rate: Original sample rate of the audio.
        audio_data: Audio data array.
        target_sr: Target sample rate (default 16000 Hz).

    Returns:
        Normalized and resampled audio data.

    Raises:
        ValueError: If sample rate is invalid or audio data has invalid shape/dtype.
    """
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    if target_sr <= 0:
        raise ValueError(f"Target sample rate must be positive, got {target_sr}")

    if not isinstance(audio_data, np.ndarray):
        raise ValueError("audio_data must be a numpy array")

    if audio_data.ndim > 2:
        raise ValueError(f"Audio data must be 1D or 2D, got {audio_data.ndim}D array")

    if np.issubdtype(audio_data.dtype, np.complexfloating):
        raise ValueError(f"Complex audio data is not supported, got dtype {audio_data.dtype}")

    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    elif audio_data.dtype == np.uint8:
        audio_data = (audio_data.astype(np.float32) - 128) / 128.0
    else:
        audio_data = audio_data.astype(np.float32)

    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    if sample_rate != target_sr:
        num_samples = int(len(audio_data) * target_sr / sample_rate)
        audio_data = signal.resample(audio_data, num_samples)

    return audio_data
