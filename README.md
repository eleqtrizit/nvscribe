# scribe — local transcription & speaker diarization on NVIDIA GPUs (Linux)

> ## Acknowledgements & gratitude
>
> **This project is a fork of
> [scribe](https://github.com/trailofbits/scribe) by
> [Trail of Bits](https://www.trailofbits.com/).** The original repo,
> its design, and the great majority of its code are entirely their
> work — without it this NVIDIA-centric variant would not exist.
>
> **Please go star and credit the original repository:
> <https://github.com/trailofbits/scribe>.** This fork only adapts that
> work to run natively on NVIDIA GPUs (including the
> NVIDIA DGX Spark), by swapping the Apple/MLX transcription backend
> for NVIDIA NeMo + Parakeet and the macOS-native diarization for the
> senko GPU stack. All credit for the overall architecture, the
> parallel-loading pipeline, the merging logic, the CLI, the tests,
> and the documentation belongs to the original authors. Thank you,
> Trail of Bits.

---

## Quick install

```sh
uv tool install https://github.com/eleqtrizit/nvscribe.git
```

Then run `scribe` from anywhere. See [Requirements](#requirements) below
for prerequisites (NVIDIA GPU, Linux, ffmpeg).

> **aarch64 / DGX Spark:** `uv tool install` from a git URL does not
> apply `[tool.uv] override-dependencies` from `pyproject.toml`, so the
> triton platform override is ignored. Use an overrides file instead:
>
> ```sh
> echo 'triton; platform_machine == "x86_64"' > overrides.txt
> uv tool install --overrides overrides.txt https://github.com/eleqtrizit/nvscribe.git
> ```

---

Local transcription and speaker diarization running entirely on
**NVIDIA GPUs** on **Linux**, built on NVIDIA's own software stack:

- **Transcription** — [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) with
  the [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
  FastConformer-TDT ASR model.
- **Diarization** — [senko](https://github.com/narcotic-sh/senko), which
  runs Pyannote segmentation and CAM++ speaker embedding models on the
  CUDA GPU via PyTorch.

> ⚠️ **Tested hardware:** This fork has been tested on the
> [NVIDIA DGX Spark](https://www.nvidia.com/en-us/dgx/dgx-spark/)
> (Grace Blackwell GB10, aarch64) and the **NVIDIA GeForce RTX 5090**
> (Blackwell GB202, x86_64). It targets NVIDIA hardware and an
> NVIDIA CUDA driver. Other NVIDIA GPUs (Ampere, Hopper, Blackwell
> datacenter cards, etc.) should work but are **untested** by the
> maintainer. Non-NVIDIA platforms (Apple Silicon, CPU-only, AMD ROCm,
> etc.) are **not supported**, and **macOS is not supported** — this
> fork requires Linux + an NVIDIA CUDA driver. Use the
> [original scribe](https://github.com/trailofbits/scribe) for macOS / Apple/MLX.

Takes any audio or video file ffmpeg can read (mp4, mp3, m4a, webm, …),
transcribes the speech with Parakeet TDT v3, optionally identifies who
spoke when using GPU diarization, and produces a speaker-attributed
transcript. Input is automatically converted to 16 kHz mono WAV for
processing. Transcription and diarization both run on the CUDA GPU, and
models download automatically on first run from Hugging Face (no account
needed).

## Requirements

- An **NVIDIA GPU** with a CUDA-capable driver. Developed and tested on
  the **NVIDIA DGX Spark** (Grace Blackwell GB10) and the **NVIDIA
  GeForce RTX 5090** (Blackwell GB202). CUDA compute
  capability ≥ 8.0 (Ampere / Hopper / Blackwell) is expected.
- **Linux** — this fork does not run on macOS or Windows. The DGX Spark
  ships Ubuntu; other Linux distros may work but are untested.
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [ffmpeg](https://ffmpeg.org/) — all input is normalized to 16 kHz mono
  WAV: `apt install ffmpeg`

## Install

```sh
uv sync
```

This installs `nemo_toolkit[asr]` (which pulls in a CUDA-enabled PyTorch
from `download.pytorch.org/whl/cu129`) plus senko. On first run the
Parakeet and diarization models are downloaded to the local Hugging Face
/ NeMo cache.

> Note for DGX Spark (aarch64): see **aarch64 / Grace Blackwell notes**
> below for why the `senko[nvidia-windows]` extra and a `triton`
> platform override are used in `pyproject.toml`.

## Usage

```sh
scribe meeting.wav
scribe recording.mp4
scribe podcast.mp3
```

Writes `meeting.txt` / `recording.txt` / `podcast.txt`:

```
SPEAKER_00: Thank you for joining. Let's start with the agenda.

SPEAKER_01: Sure, I wanted to discuss the roadmap first.
```

### Transcript only (no diarization)

```sh
scribe meeting.wav --no-diarize
```

### Options

```sh
scribe meeting.wav -o transcript.txt      # custom output path
scribe meeting.wav -o -                   # write to stdout
scribe meeting.wav --format json          # JSON with timestamps
scribe meeting.wav --format json -o -     # JSON to stdout
scribe meeting.wav --no-diarize           # transcript without speakers
```

### JSON output

```sh
scribe meeting.wav --format json
```

Writes `meeting.json` with timestamps and speaker attribution:

```json
{
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "text": "Thank you for joining.",
      "start": 0.531,
      "end": 2.874
    }
  ]
}
```

## Notes on NVIDIA hardware / memory

The Parakeet encoder runs with **local attention**
(`rel_pos_local_attn`, context 256+256) so long-form audio
(up to ~3 hours) transcribes with a small, predictable memory
footprint — important on shared GPUs like the DGX Spark. If your GPU is
busy with other workloads, transcription and diarization still run in
parallel on the same device.

## aarch64 / Grace Blackwell (DGX Spark) notes

The DGX Spark's GB10 is `aarch64`. Two packages in senko's full GPU
stack lack aarch64 wheels, so `pyproject.toml` selects workarounds:

- **`senko[nvidia-windows]`** instead of `senko[nvidia]` — this skips
  `kaldifeat` and RAPIDS (`cudf` / `cuml`), which have no aarch64
  wheels. The neural diarization models (Pyannote segmentation, CAM++)
  still run on the CUDA GPU via PyTorch; only fbank extraction and
  clustering fall back to CPU (senko's documented behavior when
  `kaldifeat` is absent).
- **`[tool.uv] override-dependencies`** scopes `triton` to `x86_64`,
  since torch's cu129 wheel requires `triton` on all Linux but triton
  has no aarch64 wheel. `triton` is only needed for `torch.compile`,
  not inference.

The CUDA-enabled torch comes from `download.pytorch.org/whl/cu129`
(`torch==2.8.0+cu129`), which ships Blackwell-capable aarch64 wheels. If
a future torch release drops or adjusts these, re-run `uv sync` to
re-resolve.

On the DGX Spark, torch may print a one-time `UserWarning` that the GB10
is CUDA capability 12.1 while this build supports up to 12.0 — CUDA
still initializes and inference runs correctly; a future `torch+cu130`
wheel will silence it. A similar one-time `onnxruntime` warning about
`/sys/class/drm/card0/device/vendor` may appear; it comes from
`silero-vad`'s onnxruntime import-time GPU probe (which has no aarch64
CUDA provider and falls back to CPU) and is harmless.

## Development

```sh
uv sync --dev
uv run pytest -q
uv run ruff check
```

## License

MIT
