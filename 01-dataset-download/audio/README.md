# Apertus Audio Data Pipeline — Status Overview

## Summary

| Category | Hours | Languages |
|---|---|---|
| **Stage 2 — Converted (audio + text)** | **586,000h** | **70+** |
| **Stage 2 — Downloaded, pending conversion** | **21,000h** | +5 new |
| **Stage 1 — Audio only, pending transcription** | **1,143,000h** | 24 |
| **Grand Total** | **1,750,000h** | **70+** |

---

## Stage 2 — Converted to SHAR (audio + text alignment)

Standardized in Lhotse SHAR format with pre-computed RMS, text tokenization, and metadata. Ready for tokenization and training.

### Transcription (audio → original language text)

| Dataset | Hours | Languages | Source |
|---|---|---|---|
| Granary YODAS (ASR) | 105,027h | 25 EU languages | YouTube CC (NVIDIA Whisper) |
| Libriheavy | 91,501h | English | LibriVox audiobooks |
| MLS | 50,687h | 8 languages (en, de, nl, fr, es, it, pt, pl) | LibriVox audiobooks |
| Legco Speech | 43,626h | Cantonese | Hong Kong Legislative Council |
| People's Speech (dirty) | 41,620h | English | Podcasts, audiobooks |
| GigaSpeech2 | 30,792h | Thai, Indonesian, Vietnamese | YouTube |
| EuroSpeech | 14,573h | 5 languages (no, it, uk, bg, fr) | European Parliament |
| People's Speech (clean) | 11,789h | English | Podcasts, audiobooks |
| SeamlessAlign | 7,420h | 5 Indian (hi, kn, ta, te, ur) | Web-crawled speech |
| Granary YTC EN | 4,295h | English | YouTube Commons |
| CommonVoice | ~3,000h | 48 languages | Crowd-sourced |
| Kathbath | 1,460h | 12 Indian languages | Read speech |
| Granary YODAS (AST) | 84,565h | 24 EU languages (non-EN) | YouTube CC (NVIDIA Whisper) |
| ParlaSpeech-RS | 755h | Serbian | Serbian Parliament |
| ParlamentParla | 746h | Catalan | Catalan Parliament |
| Coral v3 | 665h | Korean | Conversation + read aloud |
| SPC-R | 482h | Swiss German | Swiss Parliament |
| Kazakh Speech | 468h | Kazakh | ASR corpus |
| Audiocite | 310h | French | Audiobooks |
| MultiDialog | 262h | English | Spoken multi-turn dialogue |
| Zoengjyutgaai | 189h | Cantonese | Classic Chinese novels |
| Multimed | 137h | 5 languages (en, zh, fr, de, vi) | Medical speech |
| WavePulse Radio | 101h | English | US radio broadcasts |
| Zeroth Korean | 52h | Korean | Read speech |
| F1 Team Radio | 44h | English | Formula 1 team communications |
| ViMedCSS | 23h | Vietnamese | Medical speech |

### Translation (audio → English text)

| Dataset | Hours | Languages | Direction |
|---|---|---|---|
| Granary YODAS (AST translate) | 84,565h | 24 EU languages | X → English |
| SeamlessAlign (translate) | 7,420h | 5 Indian languages | X → English |

### Key Highlights

- **70+ languages** with text supervision
- **48 languages** from CommonVoice alone (including Arabic, Japanese, Turkish, Persian, Georgian, Welsh, Irish)
- **12 Indian languages** from Kathbath (Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu)
- **25 EU languages** from NVIDIA Granary with Whisper pseudo-labels
- **92,000h speech translation** data (EU + Indian languages → English)
- Unique domains: F1 radio, Swiss Parliament, Hong Kong legislature, US radio, medical speech, classic Chinese storytelling

---

## Stage 2 — Downloaded, Pending Conversion

| Dataset | Hours (est) | Languages | Notes |
|---|---|---|---|
| GigaSpeech EN | 10,000h | English | Blocked on text normalization |
| Majestrino Captions | 7,100h | Mixed | Qwen3-Omni audio descriptions |
| Omnilingual ASR | 3,000h | 300+ languages | Meta's multilingual corpus |
| Allo-AVA | 253h | English | Conversational with emotion |
| HeySQuAD | ~200h | English | Spoken question answering |
| Samromur Children | 131h | Icelandic | Children's speech |
| Uzbek Speech | ~100h | Uzbek | News speech |
| Russian LibriSpeech | 98h | Russian | Read speech |
| Linto Arabic | 93h | Tunisian Arabic | Dialectal Arabic |
| otoSpeech | 141h | English | Full-duplex speech |
| Podcast Fillers | 51h | English | Disfluency annotations |
| WildASR | ~50h | English | In-the-wild ASR |
| neuro-parakeet-food | ~50h | English | Transcribed speech |

---

## Audio-Only Data — Repurposing for Stage 2

These datasets were used in stage 1 (audio-only pretraining). For stage 2, they will be transcribed and annotated to add text supervision. During stage 2 training, loss on audio tokens is disabled — the model only learns from the text targets while retaining audio understanding from stage 1.

| Dataset | Hours | Languages | Source | Stage 2 Plan |
|---|---|---|---|---|
| VoxPopuli | 788,000h | 23 EU languages | European Parliament | Transcribe (Whisper/Canary) + Translate to English |
| Unsupervised People's Speech | 355,000h | English | Archive.org (CC-BY/CC-BY-SA) | Transcribe (Whisper/Canary) |
| Suno | 17,634h | — | AI-generated music | Annotate (Qwen3-Omni-Captioner) |
| MTG-Jamendo | 3,395h | — | Creative Commons music | Annotate (Qwen3-Omni-Captioner) |
| **Total** | **1,164,000h** | **24 languages** | | |

---

## Audio Captioning (Stage 1 → Description)

Qwen3-Omni-Captioner generated rich audio descriptions for stage 1 data. These captions describe speaker characteristics, emotion, acoustics, and recording quality — enabling the `<|audio_annotate|>` task.

| Dataset | Captions Available | Caption Quality |
|---|---|---|
| AudioSet (bal + unbal) | ~360K captions | Rich, paragraph-level descriptions |
| Gemeinderat | Available | Swiss parliamentary speech descriptions |
| Suno | Available | Music descriptions |
| MTG-Jamendo | Available | Music descriptions |

---

## Pipeline Architecture

```
Raw Data (HuggingFace / OpenSLR / Custom)
    ↓
Download (huggingface-cli / git lfs)
    ↓
SHAR Conversion (prepare_parquet_to_shar / prepare_wds_to_shar / prepare_lhotse_recipe_to_shar)
    ↓ Pre-computed: RMS dB, text tokenization, metadata
    ↓ Filtered: quiet audio < -40dB removed
Lhotse SHAR Format
    ↓
Audio Tokenization (WavTokenizer @ 40 tok/s)
    ↓
Training-Ready Indexed Dataset
```

### Task Tokens

| Token | ID | Task |
|---|---|---|
| `<\|stt_transcribe\|>` | 131082 | Audio → original language text |
| `<\|stt_translate\|>` | 131086 | Audio → English translation |
| `<\|audio_annotate\|>` | 131087 | Audio → rich description |
| `<\|stt_continue\|>` | 131083 | Interleaved audio → text continuation |
| `<\|tts_continue\|>` | 131084 | Interleaved text → audio continuation |

---

## Token Budget Estimate

At 40 audio tokens/second:

| Data | Hours | Audio Tokens | % of 1T |
|---|---|---|---|
| Stage 2 (converted) | 586,000h | 84.4B | 8.4% |
| Stage 2 (unconverted) | 21,000h | 3.0B | 0.3% |
| Stage 1 (audio only) | 1,164,000h | 167.6B | 16.8% |
| **Total audio** | **1,771,000h** | **255.0B** | **25.5%** |

*Text tokens add ~1-2% on top of audio tokens.*

---

*Last updated: April 2026*
*Pipeline: [benchmark-audio-tokenizer](https://github.com/swiss-ai/benchmark-audio-tokenizer) | Tokenizer: [apertus-omni-tokenizer](https://github.com/swiss-ai/apertus-omni-tokenizer)*
