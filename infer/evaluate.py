"""PER / CER / WER of a synthesis run.

A run directory is one of infer/mel2wav.py's output folders and ends up holding the wavs,
asr_results.json (`asr`) and the MFA re-alignment of the wavs (`align`).

    python evaluate.py asr    --run-dir $RUN   # Whisper transcripts (GPU)
    python evaluate.py align  --run-dir $RUN   # TextGrids (needs MFA)
    python evaluate.py report --run-dir $RUN   # PER, CER and WER
"""
import argparse
import json
import os
import re
import sys
from glob import glob
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tap_ets.phonemes import merge_frame_phonemes, read_frame_phonemes  # noqa: E402

DEFAULT_DATA_PATH = os.environ.get('TAP_ETS_DATA_PATH', '/workspace/data')
FRAME_RATE = 86.13
WHISPER_MODEL = 'openai/whisper-medium'
MFA_MODEL = 'english_us_arpa'


# text metrics

def normalize_text(text: str) -> str:
    text = text.lower().replace('-', ' ')
    return re.sub(r'[^a-z ]+', r'', text).strip()


def error_counts(prediction: str, target: str) -> Tuple[int, int, int, int]:
    """(char errors, chars, word errors, words) of the reference."""
    import Levenshtein

    prediction, target = normalize_text(prediction), normalize_text(target)
    char_errors = Levenshtein.distance(prediction, target)
    vocabulary = {w: chr(i) for i, w in enumerate(set(prediction.split() + target.split()))}  # word -> char
    word_errors = Levenshtein.distance(''.join(vocabulary[w] for w in prediction.split()),
                                       ''.join(vocabulary[w] for w in target.split()))
    return char_errors, max(len(target), 1), word_errors, max(len(target.split()), 1)


def error_rates(counts: Sequence[Sequence[int]]) -> Tuple[float, float]:
    """Corpus-level: total edits over total reference length."""
    char_errors, chars, word_errors, words = np.sum(np.asarray(counts, dtype=np.int64), axis=0)
    return char_errors / chars, word_errors / words


def levenshtein(reference: Sequence[int], hypothesis: Sequence[int]) -> int:
    """Edit distance between two id sequences."""
    n, m = len(reference), len(hypothesis)
    if n == 0 or m == 0:
        return max(n, m)
    previous, current = list(range(m + 1)), [0] * (m + 1)
    for i in range(1, n + 1):
        current[0] = i
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous, current = current, previous
    return previous[m]


# run directories

def run_name(run_dir: str) -> str:
    """.../tap_ets/synth/test -> tap_ets."""
    parts = os.path.normpath(run_dir).split(os.sep)
    return parts[parts.index('synth') - 1] if 'synth' in parts[1:] else parts[-1]


def load_asr_results(run_dir: str) -> List[dict]:
    path = os.path.join(run_dir, 'asr_results.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found - run `evaluate.py asr --run-dir {run_dir}` first')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# the steps

def run_asr(run_dir: str, split: str, data_path: str) -> dict:
    """Transcribe the wavs and write asr_results.json and asr.log."""
    import librosa
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    sample_rate = 16000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).to(device).eval()

    wav_paths = sorted(glob(os.path.join(run_dir, '*.wav')))
    if not wav_paths:
        raise FileNotFoundError(f'no wav files under {run_dir}')

    counts, records, log_lines = [], [], []
    for i, path in enumerate(wav_paths):
        session, idx = os.path.splitext(os.path.basename(path))[0].split('_')
        with open(os.path.join(data_path, 'silent_speech_dataset', split, 'voiced_parallel_data',
                               session, f'{idx}_info.json'), encoding='utf-8') as f:
            target = processor.tokenizer._normalize(json.load(f)['text'])

        audio, sr = sf.read(path)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        features = processor(audio, sampling_rate=sample_rate, return_tensors='pt').input_features
        with torch.no_grad():
            predicted_ids = model.generate(features.to(device))[0]
        prediction = processor.tokenizer._normalize(processor.decode(predicted_ids))

        counts.append(error_counts(prediction, target))
        cer, wer = counts[-1][0] / counts[-1][1], counts[-1][2] / counts[-1][3]
        records.append({'sess': session, 'idx': idx, 'target_text': target,
                        'prediction': prediction, 'cer': cer, 'wer': wer})
        log_lines += [f'trgt{i} {session} {idx}: {target}',
                      f'pred{i} {session} {idx}: {prediction}',
                      f'{i} sample cer: {cer * 100:.2f}, wer: {wer * 100:.2f}']

    cer, wer = error_rates(counts)
    log_lines.append(f'CER: {cer * 100:.2f}%, WER: {wer * 100:.2f}%')
    with open(os.path.join(run_dir, 'asr_results.json'), 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=4)
    with open(os.path.join(run_dir, 'asr.log'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')
    print(log_lines[-1])
    return {'name': run_name(run_dir), 'n_utt': len(records), 'cer': cer * 100, 'wer': wer * 100}


def run_align(run_dir: str, mfa_bin: str = 'mfa') -> None:
    """Re-align the synthesized speech to its own transcript with MFA."""
    import shutil
    import subprocess

    mfa_bin = shutil.which(mfa_bin) or mfa_bin
    if not os.path.exists(mfa_bin):
        raise RuntimeError(f'{mfa_bin} not found - see the MFA step in README.md')
    # MFA calls openfst binaries next to it, so its environment goes first on PATH
    env = {**os.environ, 'PATH': os.pathsep.join([os.path.dirname(mfa_bin), os.environ.get('PATH', '')])}

    corpus_dir = os.path.join(run_dir, '_mfa_corpus', 'spk1')
    out_dir = os.path.join(run_dir, '_mfa_out')
    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    records, skipped = load_asr_results(run_dir), 0
    for record in records:
        utterance = f'{record["sess"]}_{record["idx"]}'
        wav = os.path.join(run_dir, f'{utterance}.wav')
        transcript = (record.get('prediction') or '').strip()
        if not os.path.exists(wav) or not transcript:
            skipped += 1
            continue
        shutil.copy2(wav, os.path.join(corpus_dir, f'{utterance}.wav'))
        with open(os.path.join(corpus_dir, f'{utterance}.lab'), 'w', encoding='utf-8') as f:
            f.write(transcript + '\n')
    print(f'corpus ready: {corpus_dir} (skipped {skipped})')

    subprocess.run([mfa_bin, 'align', os.path.dirname(corpus_dir), MFA_MODEL, MFA_MODEL, out_dir,
                    '--single_speaker', '--clean', '--overwrite'], check=True, env=env)

    copied = 0
    for record in records:
        utterance = f'{record["sess"]}_{record["idx"]}'
        aligned = os.path.join(out_dir, 'spk1', f'{utterance}.TextGrid')
        if os.path.exists(aligned):
            shutil.copy2(aligned, os.path.join(run_dir, f'{utterance}.TextGrid'))
            copied += 1
    print(f'TextGrids copied: {copied}, missing: {len(records) - copied}')


def score_run(run_dir: str) -> dict:
    """CER and WER from an existing asr_results.json."""
    records = load_asr_results(run_dir)
    counts = [error_counts(r['prediction'], r['target_text']) for r in records]
    cer, wer = error_rates(counts)
    return {'name': run_name(run_dir), 'n_utt': len(records), 'cer': cer * 100, 'wer': wer * 100}


def phoneme_run(run_dir: str, split: str, data_path: str) -> dict:
    """PER between the merged sequences of the re-alignment and the ground-truth TextGrid."""
    records = load_asr_results(run_dir)
    total_edits = total_ref = missing = 0
    for record in records:
        session, idx = record['sess'], str(record['idx'])
        aligned_tg = os.path.join(run_dir, f'{session}_{idx}.TextGrid')
        if os.path.exists(aligned_tg):
            predicted = read_frame_phonemes(aligned_tg, FRAME_RATE)
        else:
            # MFA occasionally fails on an utterance; score it as empty rather than drop it
            missing += 1
            predicted = np.zeros(0, dtype=np.int64)
        reference = read_frame_phonemes(os.path.join(data_path, 'silent_speech_dataset', split,
                                                     'voiced_parallel_data', session,
                                                     f'{idx}_tg.TextGrid'), FRAME_RATE)

        target = merge_frame_phonemes(reference).tolist()
        total_edits += levenshtein(target, merge_frame_phonemes(predicted).tolist())
        total_ref += len(target)

    if missing:
        print(f'[warn] {run_name(run_dir)}: {missing} utterance(s) without a TextGrid')
    return {'per': total_edits / total_ref * 100 if total_ref else float('nan')}


# reporting

HEADER = f'{"run":<24} {"n":>4} {"PER":>8} {"CER":>8} {"WER":>8}'


def _format(value: Optional[float]) -> str:
    return '     -  ' if value is None or (isinstance(value, float) and np.isnan(value)) else f'{value:8.2f}'


def print_table(rows: List[dict]) -> None:
    print(HEADER)
    print('-' * len(HEADER))
    for row in rows:
        print(f'{row["name"]:<24} {row.get("n_utt", 0):>4} '
              f'{_format(row.get("per"))} {_format(row.get("cer"))} {_format(row.get("wer"))}')


# CLI

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    for name, help_text in [('asr', 'Whisper transcription of the synthesized speech'),
                            ('align', 'MFA re-alignment of the synthesized speech'),
                            ('report', 'PER, CER and WER')]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument('--run-dir', nargs='+', required=True)
        if name == 'align':
            p.add_argument('--mfa-bin', default=os.environ.get('MFA_BIN', 'mfa'))
        else:
            p.add_argument('--data-path', default=DEFAULT_DATA_PATH)
            p.add_argument('--split', default='test', choices=['train', 'dev', 'test'])

    args = parser.parse_args(argv)

    if args.cmd == 'align':
        for d in args.run_dir:
            run_align(d, args.mfa_bin)
        return 0

    if args.cmd == 'asr':
        rows = [run_asr(d, args.split, args.data_path) for d in args.run_dir]
    else:
        rows = []
        for d in args.run_dir:
            row = score_run(d)
            row.update(phoneme_run(d, args.split, args.data_path))
            rows.append(row)

    print_table(rows)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
