# sherpa-onnx

RealtimeSTT includes CPU INT8 sherpa-onnx engines for Parakeet and Moonshine.
They are useful when you want offline CPU inference without loading NeMo or
Transformers at runtime.

## Install

```bash
python -m pip install sherpa-onnx
```

## Engines

| Engine | Model bundle | Language notes |
| --- | --- | --- |
| `sherpa_onnx_parakeet` | `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` (engine default, 25 European languages) or `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` (English only, the `stt-server` / `stt-runner` CPU default) | v2 is more accurate on English; see below. |
| `sherpa_onnx_moonshine` | `sherpa-onnx-moonshine-tiny-en-int8` | English-only in the adapter. |

Aliases:

- `sherpa_parakeet`
- `parakeet_sherpa_onnx`
- `sherpa_moonshine`
- `moonshine_sherpa_onnx`

## Model Download Requirements

The three bundles below are downloaded and extracted automatically on first use
when the model is given by its bundle name (as the `stt-server` / `stt-runner`
CPU defaults do): into `download_root` when set, otherwise into
`~/.cache/realtimestt/sherpa-onnx`. Set
`transcription_engine_options={"auto_download_model": False}` to disable this.
Parakeet defaults to 4 threads and Moonshine to 1, the fastest settings measured
on an Apple M4. For other bundles, or offline machines, download
the `.tar.bz2` archives from the sherpa-onnx ASR model releases, extract them,
and pass the extracted directory as `model` or
`transcription_engine_options["model_dir"]`.

Known bundle names:

- `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2`
- `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2` (also `nvidia/parakeet-tdt-0.6b-v2`)
- `sherpa-onnx-moonshine-tiny-en-int8.tar.bz2`

Windows PowerShell example:

```powershell
New-Item -ItemType Directory -Path test-model-cache\sherpa-onnx -Force
curl.exe -L -o test-model-cache\sherpa-onnx\sherpa-onnx-moonshine-tiny-en-int8.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
python -c "import tarfile; tarfile.open(r'test-model-cache\sherpa-onnx\sherpa-onnx-moonshine-tiny-en-int8.tar.bz2', 'r:bz2').extractall(r'test-model-cache\sherpa-onnx')"
```

Linux/macOS example:

```bash
mkdir -p test-model-cache/sherpa-onnx
curl -L -o test-model-cache/sherpa-onnx/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
python -c "import tarfile; tarfile.open('test-model-cache/sherpa-onnx/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2', 'r:bz2').extractall('test-model-cache/sherpa-onnx')"
```

## Expected Model Files

Parakeet:

- `encoder.int8.onnx`
- `decoder.int8.onnx`
- `joiner.int8.onnx`
- `tokens.txt`

Moonshine Tiny:

- `preprocess.onnx`
- `encode.int8.onnx`
- `uncached_decode.int8.onnx`
- `cached_decode.int8.onnx`
- `tokens.txt`

## Basic Use

Moonshine:

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    transcription_engine="sherpa_onnx_moonshine",
    model="test-model-cache/sherpa-onnx/sherpa-onnx-moonshine-tiny-en-int8",
    device="cpu",
    language="en",
    transcription_engine_options={
        "num_threads": 2,
        "provider": "cpu",
    },
)
```

Parakeet:

```python
recorder = AudioToTextRecorder(
    transcription_engine="sherpa_onnx_parakeet",
    model="test-model-cache/sherpa-onnx/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    device="cpu",
    transcription_engine_options={
        "num_threads": 4,
        "provider": "cpu",
    },
)
```

You can also select the public model-family engine and request the sherpa-onnx
backend:

```python
recorder = AudioToTextRecorder(
    transcription_engine="parakeet",
    model="nvidia/parakeet-tdt-0.6b-v3",
    download_root="test-model-cache/sherpa-onnx",
    device="cpu",
    transcription_engine_options={
        "backend": "sherpa_onnx",
        "num_threads": 4,
    },
)
```

When `download_root` is set, known model ids resolve to the expected extracted
directory names under that root.

## Hotwords (domain vocabulary)

Parakeet has no text prompt, but sherpa-onnx can bias its beam search towards
listed phrases. Pass them as `hotwords`: a list, a comma- or newline-separated
string, or the path of a file with one phrase per line (`#` starts a comment).


```python
recorder = AudioToTextRecorder(
    transcription_engine="sherpa_onnx_parakeet",
    model="sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
    device="cpu",
    transcription_engine_options={
        "hotwords": ["JSON", "SQL", "MVVM", "Kubernetes", "GitHub Actions", "pgvector"],
    },
)
```

`stt-server` and `stt-runner` take the same list as `--hotwords "JSON, SQL, MVVM"`
or `--hotwords terms.txt`.

Setting hotwords switches decoding to `modified_beam_search` with
`modeling_unit="bpe"`, and builds the BPE vocabulary from the bundle's
`tokens.txt`. `hotwords_score` (default 1.0) sets the boost per token; 1.5
added more false matches with a large list, and 3.0 inserted listed words
that were not spoken. Beam search also capitalizes some ordinary phrases
("Death Trauma"), which WER does not count. Hotwords do not apply to Moonshine.

Keep lists to terms the model gets wrong: words it already spells right
(microservices, embeddings) and names that are everyday words (Go, Spring,
Zoom) only add false matches, so leave them out.

Measured on an Apple M4 (4 threads), 60 clips of one non-native English
speaker reading programming and résumé sentences dense with acronyms and
product names (laptop microphone, about 6.5 s each, 196 listed terms):

| Model | WER | Listed terms correct | Final p50 |
| --- | --- | --- | --- |
| Parakeet v3 int8 | 31.8% | 55% | 295 ms |
| Parakeet v3 int8 + hotwords | 24.6% | 74% | 370 ms |
| Parakeet v2 int8 | 24.5% | 62% | 296 ms |
| Parakeet v2 int8 + about 875 general software terms (none taken from these sentences) | 18.8% | 77% | 181 ms |
| Parakeet v2 int8 + those terms + 23 résumé names | 19.1% | 80% | 188 ms |
| Parakeet v2 int8 + a list of exactly the terms read | 18% | 81-83% | 200-310 ms |
| faster-whisper `small.en` int8 + term list as `initial_prompt` | 16.8% | 87% | 1964 ms |

The general software list also helped where it only partly matched: on 75 minutes
of technical sentences (danielrosehill/Tech-Sentences-For-ASR-Training) v2 WER
went from 2.34% to 2.23% and listed terms recognized from 88% to 92%, and on
the LibriSpeech sample, which has no software terms, from 3.13% to 3.04%.

## Common Options

| Option | Meaning |
| --- | --- |
| `model_dir` | Explicit extracted model directory. |
| `files` | Dictionary overriding individual file names/paths. |
| `num_threads` | CPU worker threads. |
| `provider` | ONNX Runtime provider, usually `"cpu"`. |
| `decoding_method` | sherpa-onnx decoding method, default `greedy_search`. |
| `debug` | Enables sherpa-onnx debug output. |
| `rule_fsts`, `rule_fars` | Optional text normalization resources. |
| `input_sample_rate`, `sample_rate` | Input/model sample-rate controls. |

Parakeet also supports transducer options such as `model_type`,
`max_active_paths`, `hotwords_file`, `hotwords_score`, `blank_penalty`,
`feature_dim`, `lm`, and `lm_scale`.

## Troubleshooting

- Missing file errors name the exact expected ONNX or `tokens.txt` path. Check
  that the archive was extracted, not just downloaded.
- Keep `model` pointed at the extracted directory, not the `.tar.bz2` archive.
- If latency is high, lower model size where possible, reduce realtime cadence,
  and tune `num_threads`.
- The Moonshine sherpa-onnx adapter is English-only.
