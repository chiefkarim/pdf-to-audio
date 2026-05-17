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

Convert PDF files to broadcast-quality audio via OCR + neural TTS. Runs fully offline.

Upload a PDF and get an MP3 back — no cloud APIs, no data sent anywhere.

## How it works

1. PDF pages are rasterised with **Poppler** then OCR'd with **Tesseract**
2. Extracted text is fed to **Coqui TTS** (Tacotron2-DDC, LJSpeech)
3. Audio segments are stitched with **pydub/ffmpeg** into a single MP3

## Source

Built from [github.com/chiefkarim/pdf-to-audio](https://github.com/chiefkarim/pdf-to-audio).
