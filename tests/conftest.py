import pytest
from pathlib import Path

from essentia_wrapper.audio import load_wav, normalize_audio


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    """Return the path to the fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def audio_files(fixtures_dir):
    """Return list of all test audio files."""
    return sorted(fixtures_dir.glob("*.wav"))


@pytest.fixture
def sample_audio_path(audio_files):
    """Return path to a single test audio file."""
    return audio_files[0]


@pytest.fixture
def sample_audio_normalized(sample_audio_path):
    """Load and normalize a sample audio file for testing."""
    sample_rate, audio_data = load_wav(str(sample_audio_path))
    return normalize_audio(sample_rate, audio_data)


@pytest.fixture
def sample_audio_raw(sample_audio_path):
    """Load a sample audio file without normalization."""
    return load_wav(str(sample_audio_path))


