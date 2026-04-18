# Changelog

## 2026-04-18
- New ASR engine: Anime-Whisper (litagin/anime-whisper), Japanese-only, specialized for anime / galgame speech (breaths, sighs, non-verbal sounds)
- Fix HF cache detection: aborted downloads leaving empty dirs no longer trigger false "cached" state

## 2026-03-24
- Subtitle window: auto word-wrap for long text (no more split segments), smooth height animation, pixmap render cache
- Overlay & subtitle window: position/size persistence across restarts
- Overlay: compact mode toggle animation
- Settings: removed valid-key whitelist restriction

## 2026-03-23
- Rebranded LiveTrans → LiveTranslate
- Model config: streaming toggle, structured output, context count, disable thinking (default on)
- Streaming translation display in overlay
- Prompt improvements: no alternatives, instant apply
- Repetition loop detection and user warning
- ASR engine labels: Accurate / Fast
- Changelog tab in settings panel
