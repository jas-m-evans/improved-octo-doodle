# Drum Coach MVP

A minimal local Streamlit app for fast drum practice feedback.

## What it does

- Upload a drum **audio** file (`wav`, `mp3`, `m4a`, `flac`, `ogg`) or a **video** file (`mp4`, `mov`)
- Analyze the clip for:
  - estimated tempo (BPM)
  - timing stability score
  - tempo drift between the first and second half
  - dynamics consistency
  - overall session score
- Save every session locally in SQLite
- Show session history and a progress trend chart

## Exact setup commands

From the repository root:

```bash
cd /home/runner/work/improved-octo-doodle/improved-octo-doodle/jas-m-evans/improved-octo-doodle/drum_coach_mvp
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Exact run command

```bash
cd /home/runner/work/improved-octo-doodle/improved-octo-doodle/jas-m-evans/improved-octo-doodle/drum_coach_mvp
source .venv/bin/activate
streamlit run app.py
```

## Expected first-run experience

1. Streamlit opens a local browser tab.
2. Upload a short practice clip.
3. Click **Analyze Session**.
4. The app shows metric cards, coaching tips, a saved history table, and an overall-score trend chart.
5. Session data is stored locally in `drum_coach_mvp/data/drum_coach_sessions.db`.

## Fast local validation

Run the inline deterministic self-check:

```bash
cd /home/runner/work/improved-octo-doodle/improved-octo-doodle/jas-m-evans/improved-octo-doodle/drum_coach_mvp
source .venv/bin/activate
python analysis.py --self-check
```

## ffmpeg note for video uploads

Audio files are the easiest path. For `mp4`/`mov` uploads, install `ffmpeg` if it is not already present.

macOS:

```bash
brew install ffmpeg
```

If extraction fails, the app shows a friendly error and suggests uploading audio directly.

## Known limitations

- The analysis is onset-based and works best on clear, close-mic practice recordings.
- Dense grooves, cymbal wash, or room noise can reduce onset accuracy.
- Video support depends on local `ffmpeg` availability.
- Session storage is local-only and not multi-user.

## Next 3 improvements

1. Add waveform/onset visualizations to explain why a score changed.
2. Support side-by-side comparison against a previous saved session.
3. Add pattern-specific analysis heuristics for grooves, paradiddles, and fills.
