# TAP-ETS: Time Aligned Phoneme Guiding for EMG-to-Speech Synthesis

Official implementation of *TAP-ETS*. Demo page: https://ongdyub.github.io/TAP-ETS/demo/

## ⚙️ Setup

```
conda env create -f environment.yml
conda activate tap-ets
```

PER also needs the Montreal Forced Aligner:

```
conda create -n mfa -c conda-forge montreal-forced-aligner
conda run -n mfa mfa model download acoustic english_us_arpa
conda run -n mfa mfa model download dictionary english_us_arpa
```

## 📦 Downloads

```text
EMG dataset        https://doi.org/10.5281/zenodo.4064408     ->  {data_path}/emg_data
Text alignments    https://github.com/dgaddy/silent_speech    ->  {data_path}/text_alignments
LibriSpeech MFA TextGrids                                     ->  {mfa_path}/{train,dev}-clean
tap_ets.pt, tap_refine.pt   https://github.com/ongdyub/TAP-ETS/releases/tag/v1.0   ->  data/models/
```

## **🚀 Usage**

### **1. Configuration**

Set `data_path`, `exp_path` and `mfa_path` in `configs/common.yaml`.

### **2. Preprocessing**

```
python preprocess/reassign.py      # clean and reindex into {data_path}/silent_speech_dataset
python preprocess/speech2feat.py   # 22kHz mel targets and normalizer.npz
```

### **3. Training**

```
python main.py exp_name={exp_name}          # TAP-ETS
python main_refine.py exp_name={exp_name}   # refinement model, from the LibriSpeech TextGrids
```

Checkpoints go to `{exp_path}/{exp_name}_{seed}`, keeping the best validation phoneme accuracy.

### **4. Inference**

```
python infer/prepare_guide.py --input_dir data/MONA_LISA --output data/MONA_LISA/guide.json

cd infer
./emg2wer.sh test
```

`prepare_guide.py` turns corrected phoneme files into `guide.json`; the MONA-LISA one is already there.
`emg2wer.sh` runs EMG → mel → wav → metrics into `{exp_path}/{exp_name}/synth/{split}`.
For your own weights add `ckpt=/path/to/tap_ets.ckpt refiner_ckpt=/path/to/tap_refine.ckpt`,
and set `MFA_BIN=/path/to/mfa` if `mfa` is in its own environment.

### **5. Evaluation**

```
cd infer
RUN={exp_path}/{exp_name}/synth/{split}

python evaluate.py asr    --run-dir $RUN
python evaluate.py align  --run-dir $RUN --mfa-bin /path/to/mfa
python evaluate.py report --run-dir $RUN
```

CER and WER come from the Whisper-medium transcript, PER from the MFA re-alignment of the synthesized speech.
