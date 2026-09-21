# HiFi-GAN (vendored)

Generator definition from the official HiFi-GAN implementation
(https://github.com/jik876/hifi-gan, MIT License, see `LICENSE`), used by
`tap_ets/vocoder.py` to turn synthesized mel-spectrograms into waveforms.
Only `models.py`, `env.py` and `utils_hifi.py` (renamed from `utils.py`) are
kept; `config_v1.json` is the V1 generator configuration for reference.
Training and fine-tuning scripts are available in the upstream repository.
