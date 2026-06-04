# beatdrop-outfit

Optional pre-installed AlphaRavis extension for TikTok-style beatdrop outfit-change videos.

- Default: disabled via `.pluginenv`.
- Current detector: ffmpeg/ffprobe + Python RMS heuristic; no BeatThis/BeatFirst dependency.
- Optional richer image diffs: Pillow.
- Phase C runner starts as `dry_run=true` only.

Enable locally:

```env
ALPHARAVIS_ENABLE_PLUGIN_SYSTEM=true
# plugins/beatdrop_outfit/.pluginenv
ENABLED=true
```
