"""Build the guide JSON that emg2feat.py conditions on.

Input files are <session>_<idx>_ph.txt holding an ARPAbet string (HH EH1 N D ER0 ...).

    python prepare_guide.py --input_dir ../data/MONA_LISA --output ../data/MONA_LISA/guide.json
"""
import argparse
import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tap_ets.phonemes import SIL_ID, SPECIAL_IDS, phoneme_string_to_ids  # noqa: E402
from tap_ets.released import GUIDE_JSON  # noqa: E402

FILE_PATTERN = re.compile(r'^(?P<session>\d+-\d+)_(?P<idx>\d+)(?P<suffix>.*)$')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output', default=GUIDE_JSON)
    args = parser.parse_args()

    guide = {}
    for file_name in sorted(os.listdir(args.input_dir)):
        match = FILE_PATTERN.match(file_name)
        if match is None or match.group('suffix') != '_ph.txt':
            continue
        with open(os.path.join(args.input_dir, file_name)) as f:
            content = f.read().strip()
        ids = [i for i in phoneme_string_to_ids(content) if i not in SPECIAL_IDS and i != SIL_ID]
        if not ids:
            raise ValueError(f'{file_name} produced an empty phoneme sequence')
        guide[f"{match.group('session')}_{match.group('idx')}"] = ids

    with open(args.output, 'w') as f:
        json.dump(guide, f)
    print(f'wrote {len(guide)} guide sequences to {args.output}')


if __name__ == '__main__':
    main()
