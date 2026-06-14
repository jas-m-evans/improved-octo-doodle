from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import audioread
import librosa
import numpy as np
import soundfile

PRACTICE_TYPES = ["single stroke", "doubles", "paradiddle", "groove", "fill", "other"]
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mpeg", ".mpg"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}
MIN_DURATION_SECONDS = 2.0
# Ignore intervals outside a practical drum practice tempo band: ~750 BPM max to ~40 BPM min.
MIN_VALID_IOI_SECONDS = 0.08
MAX_VALID_IOI_SECONDS = 1.5
# Linear score scales chosen so moderate CV stays usable while highly inconsistent takes fall quickly.
TIMING_CV_SCORE_SCALE = 220.0
DYNAMICS_CV_SCORE_SCALE = 180.0
# Leave a small middle buffer so one transition hit does not dominate first-vs-second-half drift.
DRIFT_FIRST_SECTION_RATIO = 0.45
DRIFT_SECOND_SECTION_RATIO = 0.55
ONSET_ENERGY_PERCENTILE_THRESHOLD = 15
TEMPO_DIRECTION_RUSHED = "rushed"
TEMPO_DIRECTION_DRAGGED = "dragged"


class AnalysisError(ValueError):
    """Raised when a practice file cannot be analyzed safely."""


def clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return float(max(minimum, min(maximum, value)))


def _extract_audio_with_ffmpeg(video_path: Path) -> Path:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise AnalysisError(
            "Video upload detected, but ffmpeg is not installed. Please upload audio directly "
            "(wav/mp3) or install ffmpeg first."
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        output_path = Path(temp_audio.name)

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "22050",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        raise AnalysisError(
            "Could not extract audio from the uploaded video. Please try a shorter clip or upload "
            "audio directly (wav/mp3)."
        )
    return output_path


def load_audio_from_path(file_path: str | Path) -> tuple[np.ndarray, int, float]:
    source_path = Path(file_path)
    if not source_path.exists() or source_path.stat().st_size == 0:
        raise AnalysisError("The uploaded file is empty. Please choose a valid audio or video clip.")

    suffix = source_path.suffix.lower()
    extracted_audio_path: Path | None = None
    load_target = source_path

    if suffix in VIDEO_SUFFIXES:
        extracted_audio_path = _extract_audio_with_ffmpeg(source_path)
        load_target = extracted_audio_path
    elif suffix not in AUDIO_SUFFIXES:
        raise AnalysisError("Unsupported file type. Please upload mp4, mov, wav, mp3, m4a, or flac.")

    try:
        audio, sample_rate = librosa.load(load_target, sr=22050, mono=True)
    except (audioread.exceptions.DecodeError, soundfile.LibsndfileError, EOFError, ValueError) as exc:
        raise AnalysisError(
            "The file could not be decoded. Please try another recording or upload audio directly."
        ) from exc
    finally:
        if extracted_audio_path is not None:
            extracted_audio_path.unlink(missing_ok=True)

    if audio.size == 0:
        raise AnalysisError("No audio data was detected in the file. Please try another recording.")

    duration_seconds = float(librosa.get_duration(y=audio, sr=sample_rate))
    if duration_seconds < MIN_DURATION_SECONDS:
        raise AnalysisError(
            f"The clip is too short ({duration_seconds:.1f}s). Please upload at least {MIN_DURATION_SECONDS:.0f} seconds."
        )

    peak = float(np.max(np.abs(audio)))
    normalized = audio / peak if peak > 0 else audio
    return normalized.astype(np.float32), sample_rate, duration_seconds


def _score_timing_stability(iois: np.ndarray) -> tuple[float, float]:
    mean_ioi = float(np.mean(iois))
    if mean_ioi <= 0:
        return 0.0, 1.0
    ioi_cv = float(np.std(iois) / mean_ioi)
    score = clamp(100.0 - (ioi_cv * TIMING_CV_SCORE_SCALE))
    return score, ioi_cv


def _score_dynamics(hit_energies: np.ndarray) -> tuple[float, float]:
    mean_energy = float(np.mean(hit_energies))
    if mean_energy <= 0:
        return 0.0, 1.0
    energy_cv = float(np.std(hit_energies) / mean_energy)
    score = clamp(100.0 - (energy_cv * DYNAMICS_CV_SCORE_SCALE))
    return score, energy_cv


def _tempo_from_iois(iois: np.ndarray) -> float:
    return float(np.median(60.0 / iois))


def _generate_tips(
    *,
    timing_score: float,
    tempo_drift_bpm: float,
    dynamics_score: float,
    estimated_bpm: float,
    target_bpm: float | None,
    practice_type: str,
) -> list[str]:
    tips: list[str] = []

    if abs(tempo_drift_bpm) >= 4:
        direction = TEMPO_DIRECTION_RUSHED if tempo_drift_bpm > 0 else TEMPO_DIRECTION_DRAGGED
        tips.append(
            f"You {direction} by about {abs(tempo_drift_bpm):.1f} BPM in the second half; retry 5-10 BPM slower with a click."
        )
    if timing_score < 75:
        tips.append("Timing stability is uneven. Loop a shorter phrase and lock in each hit against a steady subdivision.")
    if dynamics_score < 75:
        tips.append("Accent consistency is low. Practice controlled loud/soft contrast for 5 minutes before another full take.")
    if target_bpm and abs(estimated_bpm - target_bpm) >= 5:
        tips.append(
            f"Your estimated tempo landed near {estimated_bpm:.1f} BPM versus the {target_bpm:.0f} BPM target. Start closer to the target with a metronome count-in."
        )

    practice_specific = {
        "single stroke": "Keep the rebound even between hands; short 30-second bursts help clean up hand balance.",
        "doubles": "Focus on matching the second note of each double so the roll stays even instead of flattening out.",
        "paradiddle": "Bring out the natural accents in the sticking and check that the diddles stay relaxed.",
        "groove": "Check that backbeats stay centered even when the hi-hat pattern speeds up.",
        "fill": "Count the subdivision out loud once, then replay the fill with fewer notes if it still rushes.",
        "other": "Repeat the phrase three times in a row and listen for whether the pulse stays equally spaced each time.",
    }

    if len(tips) < 2:
        tips.append(practice_specific.get(practice_type, practice_specific["other"]))
    if len(tips) < 2:
        tips.append("Record another short take after a 1-minute reset and compare whether the score improves.")

    return tips[:4]


def analyze_audio_array(
    audio: np.ndarray,
    sample_rate: int,
    *,
    file_name: str,
    target_bpm: float | None = None,
    practice_type: str = "other",
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    if audio.size == 0:
        raise AnalysisError("No audio data was found in the upload.")

    if duration_seconds is None:
        duration_seconds = float(librosa.get_duration(y=audio, sr=sample_rate))
    if duration_seconds < MIN_DURATION_SECONDS:
        raise AnalysisError(
            f"The clip is too short ({duration_seconds:.1f}s). Please upload at least {MIN_DURATION_SECONDS:.0f} seconds."
        )

    onset_envelope = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        units="frames",
        backtrack=False,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=0.2,
        wait=1,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)

    if onset_times.size < 4:
        raise AnalysisError(
            "Not enough clear drum hits were detected. Try a louder recording, a longer clip, or upload cleaner audio."
        )

    iois = np.diff(onset_times)
    valid_mask = (iois >= MIN_VALID_IOI_SECONDS) & (iois <= MAX_VALID_IOI_SECONDS)
    valid_iois = iois[valid_mask]
    valid_starts = onset_times[:-1][valid_mask]
    if valid_iois.size < 3:
        raise AnalysisError(
            "The detected hits were too sparse or irregular to estimate tempo reliably. Try a steadier clip."
        )

    estimated_bpm = _tempo_from_iois(valid_iois)
    timing_score, timing_cv = _score_timing_stability(valid_iois)

    first_cutoff = duration_seconds * DRIFT_FIRST_SECTION_RATIO
    second_cutoff = duration_seconds * DRIFT_SECOND_SECTION_RATIO
    first_half_iois = valid_iois[valid_starts < first_cutoff]
    second_half_iois = valid_iois[valid_starts >= second_cutoff]
    first_half_bpm = _tempo_from_iois(first_half_iois) if first_half_iois.size >= 2 else estimated_bpm
    second_half_bpm = _tempo_from_iois(second_half_iois) if second_half_iois.size >= 2 else estimated_bpm
    tempo_drift_bpm = float(second_half_bpm - first_half_bpm)
    tempo_drift_penalty = clamp(abs(tempo_drift_bpm) * 4.0)
    tempo_drift_score = clamp(100.0 - tempo_drift_penalty)

    onset_strengths = onset_envelope[onset_frames]
    hit_energies = onset_strengths[
        onset_strengths > np.percentile(onset_strengths, ONSET_ENERGY_PERCENTILE_THRESHOLD)
    ]
    if hit_energies.size < 3:
        hit_energies = onset_strengths
    dynamics_score, dynamics_cv = _score_dynamics(hit_energies)

    overall_score = clamp((timing_score * 0.5) + (tempo_drift_score * 0.2) + (dynamics_score * 0.3))
    coaching_tips = _generate_tips(
        timing_score=timing_score,
        tempo_drift_bpm=tempo_drift_bpm,
        dynamics_score=dynamics_score,
        estimated_bpm=estimated_bpm,
        target_bpm=target_bpm,
        practice_type=practice_type,
    )

    return {
        "file_name": file_name,
        "target_bpm": float(target_bpm) if target_bpm else None,
        "practice_type": practice_type,
        "duration_seconds": round(duration_seconds, 2),
        "estimated_bpm": round(estimated_bpm, 2),
        "timing_stability_score": round(timing_score, 1),
        "timing_stability_cv": round(timing_cv, 4),
        "tempo_drift_bpm": round(tempo_drift_bpm, 2),
        "tempo_drift_score": round(tempo_drift_score, 1),
        "dynamics_consistency_score": round(dynamics_score, 1),
        "dynamics_consistency_cv": round(dynamics_cv, 4),
        "overall_score": round(overall_score, 1),
        "coaching_tips": coaching_tips,
        "score_weights": {"timing": 0.5, "drift": 0.2, "dynamics": 0.3},
        "score_rubric": {
            "timing_cv_scale": TIMING_CV_SCORE_SCALE,
            "dynamics_cv_scale": DYNAMICS_CV_SCORE_SCALE,
            "valid_ioi_seconds": [MIN_VALID_IOI_SECONDS, MAX_VALID_IOI_SECONDS],
            "drift_sections": [DRIFT_FIRST_SECTION_RATIO, DRIFT_SECOND_SECTION_RATIO],
        },
    }


def analyze_practice_file(
    file_path: str | Path,
    *,
    target_bpm: float | None = None,
    practice_type: str = "other",
) -> dict[str, Any]:
    audio, sample_rate, duration_seconds = load_audio_from_path(file_path)
    return analyze_audio_array(
        audio,
        sample_rate,
        file_name=Path(file_path).name,
        target_bpm=target_bpm,
        practice_type=practice_type,
        duration_seconds=duration_seconds,
    )


def _build_click_track(
    bpm: float,
    *,
    duration_seconds: float = 8.0,
    sample_rate: int = 22050,
    amplitude_pattern: tuple[float, ...] = (1.0, 0.8, 0.9, 0.75),
    drift_bpm: float = 0.0,
) -> np.ndarray:
    total_samples = int(duration_seconds * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)
    current_time = 0.0
    hit_index = 0

    while current_time < duration_seconds:
        progress = current_time / max(duration_seconds, 1e-6)
        instantaneous_bpm = bpm + (drift_bpm * progress)
        step_seconds = 60.0 / max(instantaneous_bpm, 1.0)
        sample_index = int(current_time * sample_rate)
        pulse = amplitude_pattern[hit_index % len(amplitude_pattern)]
        window = np.hanning(256).astype(np.float32) * pulse
        end_index = min(sample_index + window.size, total_samples)
        audio[sample_index:end_index] += window[: end_index - sample_index]
        current_time += step_seconds
        hit_index += 1

    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio


def _run_self_check() -> None:
    steady_audio = _build_click_track(120.0)
    steady_result = analyze_audio_array(
        steady_audio,
        22050,
        file_name="steady.wav",
        target_bpm=120,
        practice_type="single stroke",
        duration_seconds=8.0,
    )
    assert 110.0 <= steady_result["estimated_bpm"] <= 130.0, steady_result
    assert steady_result["timing_stability_score"] >= 85.0, steady_result
    assert abs(steady_result["tempo_drift_bpm"]) <= 4.0, steady_result
    assert steady_result["overall_score"] >= 80.0, steady_result

    drifting_audio = _build_click_track(100.0, drift_bpm=12.0)
    drifting_result = analyze_audio_array(
        drifting_audio,
        22050,
        file_name="drifting.wav",
        practice_type="groove",
        duration_seconds=8.0,
    )
    assert drifting_result["tempo_drift_bpm"] > 2.0, drifting_result
    assert len(drifting_result["coaching_tips"]) >= 2, drifting_result

    print("Self-check passed.")
    print(f"Steady clip result: {steady_result}")
    print(f"Drifting clip result: {drifting_result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a quick Drum Coach analysis self-check.")
    parser.add_argument("--self-check", action="store_true", help="Run deterministic inline assertions.")
    args = parser.parse_args()

    if args.self_check or not any(vars(args).values()):
        _run_self_check()
