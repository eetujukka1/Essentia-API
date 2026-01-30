import pytest
import numpy as np
import io
import tempfile
import os
from scipy.io import wavfile

from essentia_wrapper.audio import load_wav, load_wav_bytes, normalize_audio


class TestLoadWav:
    """Tests for load_wav function."""

    def test_load_wav_valid_file(self, tmp_path):
        """Test loading a valid WAV file."""
        wav_path = tmp_path / "test.wav"
        sample_rate = 44100
        duration = 0.1
        samples = int(sample_rate * duration)
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
        audio_data = (audio_data * 32767).astype(np.int16)
        wavfile.write(wav_path, sample_rate, audio_data)

        loaded_sr, loaded_data = load_wav(str(wav_path))

        assert loaded_sr == sample_rate
        assert len(loaded_data) == samples
        assert loaded_data.dtype == np.int16

    def test_load_wav_different_sample_rates(self, tmp_path):
        """Test loading WAV files with different sample rates."""
        for sample_rate in [8000, 16000, 22050, 44100, 48000]:
            wav_path = tmp_path / f"test_{sample_rate}.wav"
            samples = sample_rate // 10
            audio_data = np.zeros(samples, dtype=np.int16)
            wavfile.write(wav_path, sample_rate, audio_data)

            loaded_sr, _ = load_wav(str(wav_path))
            assert loaded_sr == sample_rate

    def test_load_wav_stereo(self, tmp_path):
        """Test loading a stereo WAV file."""
        wav_path = tmp_path / "stereo.wav"
        sample_rate = 44100
        samples = 1000
        audio_data = np.zeros((samples, 2), dtype=np.int16)
        audio_data[:, 0] = 1000
        audio_data[:, 1] = 2000
        wavfile.write(wav_path, sample_rate, audio_data)

        loaded_sr, loaded_data = load_wav(str(wav_path))

        assert loaded_sr == sample_rate
        assert loaded_data.shape == (samples, 2)

    def test_load_wav_invalid_path(self):
        """Test loading from non-existent path raises error."""
        with pytest.raises(FileNotFoundError):
            load_wav("/nonexistent/path/to/file.wav")

    def test_load_wav_invalid_file(self, tmp_path):
        """Test loading a non-WAV file raises error."""
        invalid_path = tmp_path / "not_a_wav.txt"
        invalid_path.write_text("This is not a WAV file")

        with pytest.raises(ValueError):
            load_wav(str(invalid_path))


class TestLoadWavBytes:
    """Tests for load_wav_bytes function."""

    def test_load_wav_bytes_valid(self):
        """Test loading WAV from valid bytes."""
        sample_rate = 44100
        samples = 1000
        audio_data = np.sin(np.linspace(0, 1, samples) * 2 * np.pi * 440)
        audio_data = (audio_data * 32767).astype(np.int16)

        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_data)
        wav_bytes = buffer.getvalue()

        loaded_sr, loaded_data = load_wav_bytes(wav_bytes)

        assert loaded_sr == sample_rate
        assert len(loaded_data) == samples
        assert loaded_data.dtype == np.int16

    def test_load_wav_bytes_stereo(self):
        """Test loading stereo WAV from bytes."""
        sample_rate = 44100
        samples = 1000
        audio_data = np.zeros((samples, 2), dtype=np.int16)

        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_data)
        wav_bytes = buffer.getvalue()

        loaded_sr, loaded_data = load_wav_bytes(wav_bytes)

        assert loaded_sr == sample_rate
        assert loaded_data.shape == (samples, 2)

    def test_load_wav_bytes_invalid(self):
        """Test loading invalid bytes raises error."""
        with pytest.raises(ValueError):
            load_wav_bytes(b"not a wav file")

    def test_load_wav_bytes_empty(self):
        """Test loading empty bytes raises error."""
        with pytest.raises(ValueError):
            load_wav_bytes(b"")


class TestNormalizeAudio:
    """Tests for normalize_audio function."""

    def test_normalize_int16(self):
        """Test normalization of int16 audio data."""
        sample_rate = 16000
        audio_data = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)

        result = normalize_audio(sample_rate, audio_data)

        assert result.dtype == np.float32
        assert np.isclose(result[0], 0.0, atol=1e-6)
        assert np.isclose(result[1], 0.5, atol=1e-4)
        assert np.isclose(result[2], -0.5, atol=1e-4)
        assert np.isclose(result[3], 1.0, atol=1e-4)
        assert np.isclose(result[4], -1.0, atol=1e-4)

    def test_normalize_int32(self):
        """Test normalization of int32 audio data."""
        sample_rate = 16000
        max_val = 2147483647
        audio_data = np.array([0, max_val // 2, -max_val // 2, max_val, -max_val - 1], dtype=np.int32)

        result = normalize_audio(sample_rate, audio_data)

        assert result.dtype == np.float32
        assert np.isclose(result[0], 0.0, atol=1e-6)
        assert np.isclose(result[3], 1.0, atol=1e-4)
        assert np.isclose(result[4], -1.0, atol=1e-4)

    def test_normalize_uint8(self):
        """Test normalization of uint8 audio data."""
        sample_rate = 16000
        audio_data = np.array([128, 0, 255, 64, 192], dtype=np.uint8)

        result = normalize_audio(sample_rate, audio_data)

        assert result.dtype == np.float32
        assert np.isclose(result[0], 0.0, atol=1e-6)
        assert np.isclose(result[1], -1.0, atol=1e-4)
        assert np.isclose(result[2], 127 / 128, atol=1e-4)

    def test_normalize_float32_passthrough(self):
        """Test that float32 data is passed through without scaling."""
        sample_rate = 16000
        audio_data = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)

        result = normalize_audio(sample_rate, audio_data)

        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, audio_data)

    def test_normalize_stereo_to_mono(self):
        """Test conversion of stereo to mono (takes first channel)."""
        sample_rate = 16000
        audio_data = np.array([
            [1000, 2000],
            [3000, 4000],
            [5000, 6000],
        ], dtype=np.int16)

        result = normalize_audio(sample_rate, audio_data)

        assert result.ndim == 1
        assert len(result) == 3
        expected = np.array([1000, 3000, 5000], dtype=np.float32) / 32768.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_resampling_downsample(self):
        """Test resampling from higher to lower sample rate."""
        original_sr = 44100
        target_sr = 16000
        duration = 0.1
        original_samples = int(original_sr * duration)
        audio_data = np.sin(np.linspace(0, 1, original_samples) * 2 * np.pi * 440).astype(np.float32)

        result = normalize_audio(original_sr, audio_data, target_sr=target_sr)

        expected_samples = int(original_samples * target_sr / original_sr)
        assert len(result) == expected_samples

    def test_normalize_resampling_upsample(self):
        """Test resampling from lower to higher sample rate."""
        original_sr = 8000
        target_sr = 16000
        original_samples = 800
        audio_data = np.zeros(original_samples, dtype=np.float32)

        result = normalize_audio(original_sr, audio_data, target_sr=target_sr)

        expected_samples = int(original_samples * target_sr / original_sr)
        assert len(result) == expected_samples

    def test_normalize_no_resampling_when_same_rate(self):
        """Test that no resampling occurs when sample rates match."""
        sample_rate = 16000
        samples = 1000
        audio_data = np.random.randn(samples).astype(np.float32)

        result = normalize_audio(sample_rate, audio_data, target_sr=sample_rate)

        assert len(result) == samples
        np.testing.assert_array_almost_equal(result, audio_data)

    def test_normalize_output_range(self):
        """Test that normalized output is in expected range for typical inputs."""
        sample_rate = 16000
        audio_data = np.array([-32768, 32767], dtype=np.int16)

        result = normalize_audio(sample_rate, audio_data)

        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_normalize_empty_audio(self):
        """Test normalization of empty audio array."""
        sample_rate = 16000
        audio_data = np.array([], dtype=np.int16)

        result = normalize_audio(sample_rate, audio_data)

        assert result.dtype == np.float32
        assert len(result) == 0

    def test_normalize_single_sample(self):
        """Test normalization of single sample."""
        sample_rate = 16000
        audio_data = np.array([16384], dtype=np.int16)

        result = normalize_audio(sample_rate, audio_data)

        assert len(result) == 1
        assert np.isclose(result[0], 0.5, atol=1e-4)

    def test_normalize_preserves_signal_shape(self):
        """Test that normalization preserves the general shape of the signal."""
        sample_rate = 16000
        t = np.linspace(0, 0.1, 1600)
        audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        result = normalize_audio(sample_rate, audio_data)

        zero_crossings_original = np.sum(np.diff(np.sign(audio_data.astype(float))) != 0)
        zero_crossings_normalized = np.sum(np.diff(np.sign(result)) != 0)
        assert zero_crossings_original == zero_crossings_normalized


class TestNormalizeAudioEdgeCases:
    """Edge case tests for normalize_audio function."""

    def test_zero_sample_rate(self):
        """Test that zero sample rate raises an error."""
        audio_data = np.zeros(1000, dtype=np.int16)
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            normalize_audio(0, audio_data)

    def test_negative_sample_rate(self):
        """Test that negative sample rate raises an error."""
        audio_data = np.zeros(1000, dtype=np.int16)
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            normalize_audio(-16000, audio_data)

    def test_zero_target_sample_rate(self):
        """Test that zero target sample rate raises an error."""
        audio_data = np.zeros(1000, dtype=np.float32)
        with pytest.raises(ValueError, match="Target sample rate must be positive"):
            normalize_audio(16000, audio_data, target_sr=0)

    def test_negative_target_sample_rate(self):
        """Test that negative target sample rate raises an error."""
        audio_data = np.zeros(1000, dtype=np.float32)
        with pytest.raises(ValueError, match="Target sample rate must be positive"):
            normalize_audio(16000, audio_data, target_sr=-16000)

    def test_invalid_audio_shape_3d(self):
        """Test that 3D audio array raises an error."""
        audio_3d = np.zeros((10, 10, 10), dtype=np.int16)
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            normalize_audio(16000, audio_3d)

    def test_none_audio_data(self):
        """Test that None audio data raises an error."""
        with pytest.raises(ValueError):
            normalize_audio(16000, None)

    def test_non_array_audio_data(self):
        """Test that non-array audio data raises an error."""
        with pytest.raises(ValueError):
            normalize_audio(16000, "not an array")

    def test_unsupported_dtype_complex(self):
        """Test that complex dtype raises an error."""
        audio_data = np.array([1+2j, 3+4j], dtype=np.complex64)
        with pytest.raises(ValueError, match="Complex audio data is not supported"):
            normalize_audio(16000, audio_data)
