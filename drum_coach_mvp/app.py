from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis import AnalysisError, PRACTICE_TYPES, analyze_practice_file
from storage import load_sessions, save_session

st.set_page_config(page_title="Drum Coach MVP", layout="wide")


@st.cache_data(show_spinner=False)
def _history_frame(practice_type: str) -> pd.DataFrame:
    return load_sessions(practice_type if practice_type != "All" else None)


@st.dialog("Analysis complete")
def _show_result_dialog(result: dict) -> None:
    st.success("Session saved. Review the key metrics and coaching tips below.")

    metric_columns = st.columns(5)
    metric_columns[0].metric("Estimated BPM", f"{result['estimated_bpm']:.1f}")
    metric_columns[1].metric("Timing stability", f"{result['timing_stability_score']:.1f}/100")
    metric_columns[2].metric("Tempo drift", f"{result['tempo_drift_bpm']:+.1f} BPM")
    metric_columns[3].metric("Dynamics consistency", f"{result['dynamics_consistency_score']:.1f}/100")
    metric_columns[4].metric("Overall score", f"{result['overall_score']:.1f}/100")

    st.subheader("Coaching tips")
    for tip in result["coaching_tips"]:
        st.info(tip)


def _history_section() -> None:
    st.subheader("Session history")
    filter_options = ["All", *PRACTICE_TYPES]
    selected_filter = st.selectbox("Filter history by practice type", filter_options, index=0)
    history = _history_frame(selected_filter)

    if history.empty:
        st.caption("No saved sessions yet. Analyze your first clip to start tracking progress.")
        return

    chart_data = history.sort_values("created_at")[["created_at", "overall_score"]].set_index("created_at")
    st.line_chart(chart_data, height=240)

    display_history = history.copy()
    display_history["created_at"] = display_history["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    display_history["coaching_tips"] = display_history["coaching_tips"].apply(lambda tips: " | ".join(tips))
    st.dataframe(display_history, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("Drum Coach MVP")
    st.write(
        "Upload a drum practice recording to get fast, automated feedback based on tempo consistency, drift, and hit dynamics. "
        "This MVP is best for clear practice clips and should be treated as a coaching aid, not a perfect performance judge."
    )

    with st.expander("How to interpret score", expanded=True):
        st.markdown(
            "- **80-100:** strong consistency for a quick take\n"
            "- **60-79:** useful baseline with one or two clear weaknesses to isolate\n"
            "- **Below 60:** re-record with a shorter phrase, cleaner audio, or a slower target tempo\n\n"
            "Overall score = 50% timing stability + 20% tempo drift control + 30% dynamics consistency."
        )

    upload_col, config_col = st.columns([1.4, 1])
    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload drum practice audio/video",
            type=["mp4", "mov", "wav", "mp3", "m4a", "flac", "ogg"],
            help="Video uploads require ffmpeg for audio extraction. Audio uploads are the simplest path.",
        )
    with config_col:
        target_bpm_value = st.number_input("Target BPM (optional, 0 = none)", min_value=0, max_value=400, value=0, step=1)
        practice_type = st.selectbox("Practice type", PRACTICE_TYPES, index=0)

    analyze_clicked = st.button("Analyze Session", type="primary", use_container_width=True)

    if analyze_clicked:
        if uploaded_file is None:
            st.warning("Please upload an audio or video file before analyzing.")
        else:
            suffix = Path(uploaded_file.name).suffix or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = Path(temp_file.name)

            try:
                with st.spinner("Analyzing session..."):
                    result = analyze_practice_file(
                        temp_path,
                        target_bpm=float(target_bpm_value) if target_bpm_value > 0 else None,
                        practice_type=practice_type,
                    )
                    save_session(result)
                    _history_frame.clear()
                _show_result_dialog(result)
            except AnalysisError as exc:
                st.error(str(exc))
            except (OSError, RuntimeError, sqlite3.Error) as exc:
                st.error(f"Unexpected error while analyzing the session: {exc}")
            finally:
                temp_path.unlink(missing_ok=True)

    st.divider()
    _history_section()


if __name__ == "__main__":
    main()
