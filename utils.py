STR_CLIP_ID = 'clip_id'
STR_AUDIO_SIGNAL = 'audio_signal'
STR_TARGET_VECTOR = 'target_vector'
STR_CH_FIRST = 'channels_first'
STR_CH_LAST = 'channels_last'
SPEECH_SAMPLE_RATE = 16000
import io
import os
import tqdm
import logging
import subprocess
from typing import Tuple
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

import itertools
from numpy.fft import irfft

def _resample_load_ffmpeg(path: str, sample_rate: int, downmix_to_mono: bool) -> Tuple[np.ndarray, int]:
    """
    Decoding, downmixing, and downsampling by librosa.
    Returns a channel-first audio signal.

    Args:
        path:
        sample_rate:
        downmix_to_mono:

    Returns:
        (audio signal, sample rate)
    """

    def _decode_resample_by_ffmpeg(filename, sr):
        """decode, downmix, and resample audio file"""
        channel_cmd = '-ac 1 ' if downmix_to_mono else ''  # downmixing option
        resampling_cmd = f'-ar {str(sr)}' if sr else ''  # downsampling option
        cmd = f"ffmpeg -i \"{filename}\" {channel_cmd} {resampling_cmd} -f wav -"
        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        return out

    src, sr = sf.read(io.BytesIO(_decode_resample_by_ffmpeg(path, sr=sample_rate)))
    return src.T, sr


def _resample_load_librosa(path: str, sample_rate: int, downmix_to_mono: bool, **kwargs) -> Tuple[np.ndarray, int]:
    """
    Decoding, downmixing, and downsampling by librosa.
    Returns a channel-first audio signal.
    """
    src, sr = librosa.load(path, sr=sample_rate, mono=downmix_to_mono, **kwargs)
    return src, sr


def load_audio(
    path: str or Path,
    ch_format: str,
    sample_rate: int = None,
    downmix_to_mono: bool = False,
    resample_by: str = 'ffmpeg',
    **kwargs,
) -> Tuple[np.ndarray, int]:
    """A wrapper of librosa.load that:
        - forces the returned audio to be 2-dim,
        - defaults to sr=None, and
        - defaults to downmix_to_mono=False.

    The audio decoding is done by `audioread` or `soundfile` package and ultimately, often by ffmpeg.
    The resampling is done by `librosa`'s child package `resampy`.

    Args:
        path: audio file path
        ch_format: one of 'channels_first' or 'channels_last'
        sample_rate: target sampling rate. if None, use the rate of the audio file
        downmix_to_mono:
        resample_by (str): 'librosa' or 'ffmpeg'. it decides backend for audio decoding and resampling.
        **kwargs: keyword args for librosa.load - offset, duration, dtype, res_type.

    Returns:
        (audio, sr) tuple
    """
    if ch_format not in (STR_CH_FIRST, STR_CH_LAST):
        raise ValueError(f'ch_format is wrong here -> {ch_format}')

    if os.stat(path).st_size > 4000:
        if resample_by == 'librosa':
            src, sr = _resample_load_librosa(path, sample_rate, downmix_to_mono, **kwargs)
        elif resample_by == 'ffmpeg':
            src, sr = _resample_load_ffmpeg(path, sample_rate, downmix_to_mono)
        else:
            raise NotImplementedError(f'resample_by: "{resample_by}" is not supposred yet')
    else:
        raise InvalidAudioError('Given audio is too short!')

    if src.ndim == 1:
        src = np.expand_dims(src, axis=0)
    # now always 2d and channels_first

    if ch_format == STR_CH_FIRST:
        return src, sr
    else:
        return src.T, sr