"""
Phone Cough Classifier - Audio Processing Utilities
Handles format conversion, normalization, and preprocessing
"""
import logging
from pathlib import Path
from typing import Optional, Tuple
import tempfile
import subprocess

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Audio preprocessing and format conversion"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._librosa = None
        self._sf = None
    
    @property
    def librosa(self):
        if self._librosa is None:
            import librosa
            self._librosa = librosa
        return self._librosa
    
    @property
    def soundfile(self):
        if self._sf is None:
            import soundfile as sf
            self._sf = sf
        return self._sf
    
    def convert_to_wav(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Convert audio file to WAV format (16kHz mono).
        Supports: MP3, OGG, WebM, M4A, FLAC, etc.
        """
        input_path = Path(input_path)
        
        if output_path:
            output_path = Path(output_path)
        else:
            output_path = input_path.with_suffix(".wav")
        
        try:
            # Try using pydub (handles most formats)
            from pydub import AudioSegment
            
            audio = AudioSegment.from_file(str(input_path))
            
            # Convert to mono, resample
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(self.sample_rate)
            
            # Export as WAV
            audio.export(str(output_path), format="wav")
            
            logger.debug(f"Converted {input_path} to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.warning(f"pydub conversion failed: {e}, trying ffmpeg...")
            
            # Fallback to ffmpeg
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(input_path),
                    "-ar", str(self.sample_rate),
                    "-ac", "1",
                    str(output_path)
                ], check=True, capture_output=True)
                
                return str(output_path)
                
            except Exception as e2:
                logger.error(f"FFmpeg conversion failed: {e2}")
                raise
    
    def load_and_normalize(
        self,
        audio_path: str,
        target_db: float = -20.0,
        duration: Optional[float] = None
    ) -> Tuple:
        """Load audio and normalize to target dB level"""
        import numpy as np
        
        # Load audio
        y, sr = self.librosa.load(
            audio_path,
            sr=self.sample_rate,
            duration=duration,
            mono=True
        )
        
        # Normalize to target dB
        current_db = 20 * np.log10(np.max(np.abs(y)) + 1e-10)
        gain = 10 ** ((target_db - current_db) / 20)
        y_normalized = y * gain
        
        # Clip to prevent distortion
        y_normalized = np.clip(y_normalized, -1.0, 1.0)
        
        return y_normalized, sr
    
    def trim_silence(
        self,
        y,
        top_db: int = 30
    ):
        """Trim silence from beginning and end"""
        import numpy as np
        
        y_trimmed, index = self.librosa.effects.trim(y, top_db=top_db)
        
        return y_trimmed
    
    def detect_cough_segments(
        self,
        y,
        sr: int = 16000,
        threshold_db: float = -40,
        min_duration: float = 0.1,
        max_duration: float = 2.0
    ) -> list:
        """
        Detect cough segments in audio.
        Returns list of (start_time, end_time) tuples.
        """
        import numpy as np
        
        # Compute RMS energy
        hop_length = 512
        rms = self.librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Convert to dB
        rms_db = 20 * np.log10(rms + 1e-10)
        
        # Find segments above threshold
        is_sound = rms_db > threshold_db
        
        # Convert frames to times
        frame_times = self.librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=hop_length
        )
        
        # Find segment boundaries
        segments = []
        in_segment = False
        start_time = 0
        
        for i, (is_s, t) in enumerate(zip(is_sound, frame_times)):
            if is_s and not in_segment:
                start_time = t
                in_segment = True
            elif not is_s and in_segment:
                duration = t - start_time
                if min_duration <= duration <= max_duration:
                    segments.append((start_time, t))
                in_segment = False
        
        # Handle case where audio ends during a segment
        if in_segment:
            duration = frame_times[-1] - start_time
            if min_duration <= duration <= max_duration:
                segments.append((start_time, frame_times[-1]))
        
        return segments
    
    def save_audio(
        self,
        y,
        output_path: str,
        sample_rate: Optional[int] = None
    ):
        """Save audio array to file"""
        sr = sample_rate or self.sample_rate
        self.soundfile.write(output_path, y, sr)
        logger.debug(f"Saved audio to {output_path}")
    
    def get_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        return self.librosa.get_duration(path=audio_path)


# Singleton instance
_processor = None


def get_audio_processor() -> AudioProcessor:
    """Get singleton audio processor"""
    global _processor
    if _processor is None:
        _processor = AudioProcessor()
    return _processor
