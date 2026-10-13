"""
AI-based speech/audio restoration: noise, reverberation, low resolution
(bandwidth limiting) and clipping, all handled by a single pretrained
model — VoiceFixer (Haohe Liu, MIT license,
https://github.com/haoheliu/voicefixer).

Deliberately CPU-only: VoiceFixer is plain PyTorch with no ONNX export
available, and this machine's torch build has no GPU backend for an AMD
card anyway (DirectML is only wired up for the ONNX Runtime models used by
the other tabs). Measured on this project's dev CPU at ~1.4x real-time per
30s segment, which is acceptable for an offline batch tool — not real-time,
but nowhere near impractical.

Optimized for speech/dialogue; on music-heavy soundtracks it can introduce
artifacts, since that's outside what the model was trained on.
"""

import os
import logging
import urllib.request
from typing import Callable, Optional

logger = logging.getLogger("RadeonVideoAI.AudioModels")

SAMPLE_RATE = 44100


def _wrap_urlretrieve_with_progress(progress_cb):
    """
    VoiceFixer downloads its two pretrained checkpoints (~600MB total) via
    a bare urllib.request.urlretrieve() call with no progress hook, the
    moment `import voicefixer` first runs. Monkeypatching the module-level
    function (rather than editing the third-party package) lets the status
    bar show a real download percentage instead of an indefinite "please
    wait", matching the progress UX already used for the other tabs'
    pretrained weight downloads.
    """
    real_urlretrieve = urllib.request.urlretrieve

    def wrapped(url, filename, *args, **kwargs):
        def _reporthook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = min(block_num * block_size, total_size)
                progress_cb(downloaded, total_size, os.path.basename(filename))
        return real_urlretrieve(url, filename, reporthook=_reporthook)

    return wrapped, real_urlretrieve


class AudioRestorationModel:
    """Thin wrapper around voicefixer.VoiceFixer exposing a single-chunk
    restore call, so the engine can drive progress/pause/cancel between
    chunks instead of handing it a whole multi-minute file at once."""

    def __init__(self, progress_cb: Optional[Callable[[int, int, str], None]] = None):
        if progress_cb:
            wrapped, original = _wrap_urlretrieve_with_progress(progress_cb)
            urllib.request.urlretrieve = wrapped
            try:
                import voicefixer
                self._vf = voicefixer.VoiceFixer()
            finally:
                urllib.request.urlretrieve = original
        else:
            import voicefixer
            self._vf = voicefixer.VoiceFixer()

    @property
    def active_backend(self) -> str:
        return "CPU (PyTorch) — VoiceFixer (ruido, reverberación, baja resolución, clipping)"

    def restore_chunk(self, wav_chunk, mode: int = 0):
        """
        Restores one mono float32 numpy chunk at 44.1kHz. VoiceFixer's own
        restore_inmem() already internally caps each inference pass at 30s
        of audio (its `seg_length`); feeding it chunks at or below that size
        one at a time (rather than a whole file) is what lets the engine
        report per-chunk progress and honor pause/cancel between calls.
        """
        return self._vf.restore_inmem(wav_chunk, cuda=False, mode=mode)
