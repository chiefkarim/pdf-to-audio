---
title: Pdf To Audio
emoji: 🎧
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# PDF-to-Audio

Convert PDF files to broadcast-quality audio via OCR + neural TTS. Runs fully offline — no cloud APIs, no data sent anywhere.

## How it works

1. **PDF extraction** — text layer read directly with PyMuPDF (milliseconds); Tesseract OCR fallback for scanned/image-only pages
2. **TTS synthesis** — two modes selectable per conversion:
   - **Fast** — Meta MMS-TTS (`facebook/mms-tts-eng`, VITS, 16 kHz) — ~10× real-time on CPU
   - **Quality** — Coqui Tacotron2-DDC (LJSpeech, 22 kHz) — richer prosody
3. **Audio processing** — pedalboard DSP chain (HPF, presence EQ, de-esser, compressor) applied to all output regardless of mode
4. **Export** — MP3 (192 kbps) or WAV (16-bit PCM)

Both models are pre-baked into the image — no downloads at runtime.

## Source

Built from [github.com/chiefkarim/pdf-to-audio](https://github.com/chiefkarim/pdf-to-audio).
