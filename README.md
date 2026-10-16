# XTTSv2-Finetuning-for-Tamil

A collection of scripts and instructions for fine-tuning XTTS-v2 for Tamil language text-to-speech using the Coqui TTS framework.

## Upstream Projects

This project builds upon the work of:

- **XTTSv2 Fine-Tuning for New Languages:** https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages
- **Coqui TTS:** https://github.com/coqui-ai/TTS

The Vietnamese fine-tuning repository provides the training workflow and bundled Coqui source used by the scripts. Coqui TTS provides the underlying text-to-speech framework and XTTS implementation.

Please refer to the upstream repositories for their original implementations, documentation, and applicable licenses.

## Project Structure

```text
├──src/
  ├── train_gpt_xtts.py
  ├── extend_vocab_config.py
  ├── xtts_model_inference.py
├── requirements.txt
├── README.md
├── .gitignore                    
```

## Requirements

- Python 3.10
- Conda or another Python environment manager
- PyTorch and torchaudio versions compatible with your machine
- The Coqui TTS source and training dependencies expected by the upstream fine-tuning repository
- Sufficient storage for datasets, pretrained weights, and training checkpoints

Training support depends on your hardware and the compatibility of the installed PyTorch, Coqui TTS, and Transformers versions.

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/nikitasaravanan/XTTSv2-Finetuning-for-Tamil.git
cd XTTSv2-Finetuning-for-Tamil
```

Replace the placeholders with your actual GitHub URL and repository directory.

### 2. Obtain the upstream fine-tuning repository

```bash
git clone https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages.git
```

The upstream repository contains the `TTS/` source tree expected by the training scripts.

### 3. Create and activate a Python environment

For Conda:

```bash
conda create -n xtts-tamil python=3.10
conda activate xtts-tamil
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Install or expose the upstream source according to its installation instructions. If you keep the upstream repository as a sibling directory, you can expose its source for the current terminal session on macOS or Linux:

```bash
export PYTHONPATH="../XTTSv2-Finetuning-for-New-Languages${PYTHONPATH:+:$PYTHONPATH}"
```

Run this command from your own repository's root. Adjust the path if you store the upstream repository elsewhere.

Check that the imports resolve before training:

```bash
python -c "from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer; from trainer import Trainer; print('Imports successful')"
```

If this fails, verify the upstream directory path and dependency versions before starting a training run.

## Dataset Preparation

Prepare one or more datasets using this structure:

```text
datasets-1/
├── wavs/
│   ├── sample_001.wav
│   └── sample_002.wav
├── metadata_train.csv
└── metadata_eval.csv
```

Each metadata file should use pipe-separated columns:

```text
audio_file|text|speaker_name
wavs/sample_001.wav|Example Tamil sentence.|speaker_1
wavs/sample_002.wav|Another example sentence.|speaker_1
```

Requirements:

- Audio paths must resolve correctly relative to the dataset root.
- Audio filenames in the metadata must include their actual file extensions.
- Transcriptions should accurately match the recordings.
- Training and evaluation files should contain separate examples.
- Use `ta` as the language code for Tamil.

## Training

The GPT fine-tuning script supports multiple datasets using the `--metadatas` argument.

Example command for a Mac:

```bash
python train_gpt_xtts.py \
  --output_path checkpoints/ \
  --metadatas datasets-1/metadata_train.csv,datasets-1/metadata_eval.csv,vi datasets-2/metadata_train.csv,datasets-2/metadata_eval.csv,ta \
  --num_epochs 5 \
  --batch_size 8 \
  --grad_acumm 4 \
  --max_text_length 400 \
  --max_audio_length 330750 \
  --weight_decay 1e-2 \
  --lr 5e-6 \
  --save_step 50000
```

Adjust dataset paths and training parameters to match your setup.

### macOS notes

- Do not set `CUDA_VISIBLE_DEVICES=0` on a Mac using Apple Silicon.
- Apple MPS availability does not guarantee that the training framework supports the complete training pipeline on MPS.
- Start with a smaller batch size and fewer data-loader workers when checking compatibility.
- Test a short run before committing to a long training session.
- Monitor memory use and confirm that the model and tensors are actually using the intended device.

## Vocabulary Extension

Use `extend_vocab_config.py` to extend the tokenizer vocabulary and prepare the configuration before fine-tuning.

Check the script's command-line arguments and the upstream instructions for the required pretrained files, metadata paths, language code, and vocabulary size.

Do not assume the original XTTS-v2 checkpoint already supports Tamil natively.

## Inference

Use `xtts_model_inference.py` to run inference with the fine-tuned model.

The script requires the relevant trained checkpoint, model configuration, vocabulary files, and reference audio where applicable.

Update local paths and select the intended language code in your inference configuration.

## Checkpoints and Generated Files

Large files are intentionally excluded from version control, including:

- Pretrained model weights
- Fine-tuned checkpoints
- Tokenizer and vocabulary artifacts, where redistribution is not permitted
- Training logs and generated audio
- Local datasets

Follow the upstream instructions to obtain the required model files. Check the applicable model license before distributing checkpoints or generated assets.

## Reproducibility

For reproducible results, record:

- Python version
- PyTorch and torchaudio versions
- Transformers and tokenizers versions
- The commit hash of the upstream fine-tuning repository
- Dataset sources and preprocessing steps
- Training parameters and random seeds, where applicable

## Licensing and Attribution

This repository references and builds upon external projects. Refer to the upstream repositories and model documentation for applicable code and model licenses.

The Coqui TTS code license does not automatically determine the license for pretrained XTTS-v2 weights, datasets, or fine-tuned model checkpoints. Review each applicable license before redistribution or commercial use.

## Acknowledgements

Thanks to the authors and contributors of the XTTS-v2 fine-tuning guide and the Coqui TTS project for making their work available.
