"""Clean and reindex the Gaddy dataset into <data_path>/silent_speech_dataset/{train,dev,test}.

Utterances without an alignment and boundary clips are dropped, sessions are reindexed from 0,
and a silent recording gets the index of its paired voiced one. Based on read_emg.py of
https://github.com/dgaddy/silent_speech.

Expects <data_path>/emg_data/ and <data_path>/text_alignments/.
"""
import json
import os
import re
import shutil

import hydra
from omegaconf import DictConfig

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class EMGDirectory:
    def __init__(self, session_index, directory, silent):
        self.session_index = session_index
        self.directory = directory
        self.silent = silent

    def __lt__(self, other):
        return self.session_index < other.session_index

    def __repr__(self):
        return self.directory


def split_path(path):
    parts = path.split('/')
    return parts[-3], parts[-2], parts[-1].split('_')[0]  # kind, session, idx


class Matcher:
    """Examples of one split, plus the voiced recording of every sentence."""

    def __init__(self, emg_dir, alignment_dir, testset_file, dev=False, test=False):
        with open(testset_file) as f:
            testset_json = json.load(f)
        devset, testset = testset_json['dev'], testset_json['test']

        directories = []
        for kind, silent in [('silent_parallel_data', True), ('nonparallel_data', False), ('voiced_parallel_data', False)]:
            for session_dir in sorted(os.listdir(os.path.join(emg_dir, kind))):
                directories.append(EMGDirectory(len(directories), os.path.join(emg_dir, kind, session_dir), silent))

        self.example_indices = []
        self.voiced_data_locations = {}  # (book, sentence_index) -> (directory, idx)
        for directory in directories:
            for fname in os.listdir(directory.directory):
                match = re.match(r'(\d+)_info.json', fname)
                if match is None:
                    continue
                idx = int(match.group(1))
                json_path = os.path.join(directory.directory, fname)
                with open(json_path) as f:
                    info = json.load(f)
                if info['sentence_index'] < 0 or info['text'] == '.':  # boundary clips are marked -1
                    continue
                if not directory.silent:
                    _, session, idx_str = split_path(json_path)
                    tg_path = os.path.join(alignment_dir, session, f'{session}_{idx_str}_audio.TextGrid')
                    if not os.path.exists(tg_path):
                        print(f'missing alignment {tg_path}: {info["text"]}')
                        continue
                location = [info['book'], info['sentence_index']]
                in_test, in_dev = location in testset, location in devset
                if (test and in_test) or (dev and in_dev) or (not test and not dev and not in_test and not in_dev):
                    self.example_indices.append((directory, idx))
                if not directory.silent:
                    self.voiced_data_locations[tuple(location)] = (directory, idx)
        self.example_indices.sort()
        self.emg_dir = emg_dir

    def match(self, kind, session, idx):
        """(kind, session, idx) of the voiced recording paired with a silent one."""
        with open(os.path.join(self.emg_dir, kind, session, f'{idx}_info.json')) as f:
            info = json.load(f)
        location = (info['book'], info['sentence_index'])
        if location not in self.voiced_data_locations:
            print(f'no voiced recording for {location} ({kind}/{session}/{idx})')
            return None, None, None
        directory, idx_voiced = self.voiced_data_locations[location]
        parts = directory.directory.split('/')
        return parts[-2], parts[-1], idx_voiced


def copy_example(old_dir, old_idx, new_dir, new_idx):
    os.makedirs(new_dir, exist_ok=True)
    for suffix in ['audio_clean.flac', 'emg.npy', 'info.json']:
        shutil.copyfile(os.path.join(old_dir, f'{old_idx}_{suffix}'), os.path.join(new_dir, f'{new_idx}_{suffix}'))


def copy_textgrid(alignment_dir, session, old_idx, new_dir, new_idx):
    shutil.copyfile(os.path.join(alignment_dir, session, f'{session}_{old_idx}_audio.TextGrid'),
                    os.path.join(new_dir, f'{new_idx}_tg.TextGrid'))


@hydra.main(version_base=None, config_path='../configs', config_name='common')
def main(cfg: DictConfig):
    emg_dir = os.path.join(cfg.data_path, 'emg_data')
    alignment_dir = os.path.join(cfg.data_path, 'text_alignments')
    output_dir = os.path.join(cfg.data_path, 'silent_speech_dataset')
    testset_file = os.path.join(PROJECT_DIR, 'testset_largedev.json')

    counters = {}
    for split in ['train', 'dev', 'test']:
        matcher = Matcher(emg_dir, alignment_dir, testset_file, dev=split == 'dev', test=split == 'test')
        print(f'{split} dataset: {len(matcher.example_indices)} examples')
        for directory, index in matcher.example_indices:
            kind, session, idx = split_path(os.path.join(directory.directory, f'{index}_emg.npy'))
            if kind == 'voiced_parallel_data':
                continue  # copied with its silent counterpart
            key = f'{split}/{kind}/{session}'
            new_idx = counters.setdefault(key, 0)
            if kind == 'silent_parallel_data':
                kind_v, session_v, idx_v = matcher.match(kind, session, idx)
                if kind_v is None:
                    continue
                copy_example(directory.directory, idx, os.path.join(output_dir, split, kind, session_v), new_idx)
                voiced_dir = os.path.join(output_dir, split, kind_v, session_v)
                copy_example(os.path.join(emg_dir, kind_v, session_v), idx_v, voiced_dir, new_idx)
                copy_textgrid(alignment_dir, session_v, idx_v, voiced_dir, new_idx)
            else:  # nonparallel_data
                new_dir = os.path.join(output_dir, split, kind, session)
                copy_example(directory.directory, idx, new_dir, new_idx)
                copy_textgrid(alignment_dir, session, idx, new_dir, new_idx)
            counters[key] += 1

    for key, count in counters.items():
        print(key, count)


if __name__ == '__main__':
    main()
