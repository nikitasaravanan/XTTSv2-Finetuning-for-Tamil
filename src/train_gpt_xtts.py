import os
import gc
import torch
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

import argparse

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

import psutil, os
import threading
import time
import psutil
import os

def monitor_resources(interval=50):
    process = psutil.Process(os.getpid())
    while True:
        mem = process.memory_info().rss / 1024 ** 2  # MB
        statvfs = os.statvfs('/')
        total = statvfs.f_frsize * statvfs.f_blocks / (1024**3)  # GB
        available = statvfs.f_frsize * statvfs.f_bavail / (1024**3)  # GB
        used = total - available

        gpu_info = ""
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2
            gpu_mem_cached = torch.cuda.memory_reserved() / 1024 ** 2
            gpu_info = f"| GPU Allocated: {gpu_mem_allocated:.2f} MB | Cached: {gpu_mem_cached:.2f} MB"
        elif torch.backends.mps.is_available():
            gpu_info = "| MPS backend detected, GPU mem stats not supported."

        print(f"[RESOURCE MONITOR] RSS Memory: {mem:.2f} MB | Disk Used: {used:.2f} GB | Disk Free: {available:.2f} GB / {total:.2f} GB {gpu_info}")
        time.sleep(interval)

def print_memory_usage(step_desc=""):
    import psutil
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss = mem_info.rss / 1024 ** 2  # in MB
    statvfs = os.statvfs('/')
    total = statvfs.f_frsize * statvfs.f_blocks / (1024**3)
    available = statvfs.f_frsize * statvfs.f_bavail / (1024**3)
    used = total - available
    if torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2
        gpu_mem_cached = torch.cuda.memory_reserved() / 1024 ** 2
        print(f"[GPU MEMORY] {step_desc} | Allocated: {gpu_mem_allocated:.2f} MB | Cached: {gpu_mem_cached:.2f} MB")

    print(f"[DISK USAGE] {step_desc} | Used: {used:.2f} GB | Free: {available:.2f} GB / Total: {total:.2f} GB")

    print(f"[CPU MEMORY] {step_desc} | RSS: {rss:.2f} MB")

    if torch.backends.mps.is_available():
        print(f"[MPS MEMORY] {step_desc} | Memory usage stats not supported on MPS backend.")


def create_xtts_trainer_parser():
    parser = argparse.ArgumentParser(description="Arguments for XTTS Trainer")

    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to pretrained + checkpoint model")
    parser.add_argument("--metadatas", nargs='+', type=str, required=True,
                        help="train_csv_path,eval_csv_path,language")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Mini batch size")
    parser.add_argument("--grad_acumm", type=int, default=1,
                        help="Grad accumulation steps")
    parser.add_argument("--max_audio_length", type=int, default=255995,
                        help="Max audio length")
    parser.add_argument("--max_text_length", type=int, default=200,
                        help="Max text length")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=0.0, 
                        help="Maximum norm for gradient clipping. Set to 0 to disable clipping.")
    parser.add_argument("--save_step", type=int, default=5000,
                        help="Save step")
    
    parser.add_argument("--gpt_checkpoint", type=str, default=None,
                        help="Path to GPT checkpoint to resume from")

    parser.add_argument("--xtts_checkpoint", type=str, default=None,
                        help="Path to XTTS checkpoint to resume from")
    
    parser.add_argument("--start_epoch", type=int, default=0,
                    help="Epoch to start from when loading model weights only.")
    
    return parser



def train_gpt(metadatas, num_epochs, batch_size, grad_acumm, output_path, max_audio_length, max_text_length, lr, weight_decay, save_step, gpt_checkpoint=None, start_epoch=0):
    #  Logging parameters
    RUN_NAME = "GPT_XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None

    # Set here the path that the checkpoints will be saved. Default: ./run/training/
    # OUT_PATH = os.path.join(output_path, "run", "training")
    OUT_PATH = output_path

    # Training Parameters
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
    START_WITH_EVAL = False  # if True it will star with evaluation
    BATCH_SIZE = batch_size  # set here the batch size
    GRAD_ACUMM_STEPS = grad_acumm  # set here the grad accumulation steps


    # Define here the dataset that you want to use for the fine-tuning on.
    DATASETS_CONFIG_LIST = []
    for metadata in metadatas:
        train_csv, eval_csv, language = metadata.split(",")
        print(train_csv, eval_csv, language)

        config_dataset = BaseDatasetConfig(
            formatter="coqui",
            dataset_name="ft_dataset",
            path=os.path.dirname(train_csv),
            meta_file_train=os.path.basename(train_csv),
            meta_file_val=os.path.basename(eval_csv),
            language=language,
        )

        DATASETS_CONFIG_LIST.append(config_dataset)

    # Define the path where XTTS v2.0.1 files will be downloaded
    CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)


    # DVAE files
    DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

    # Set the path to the downloaded files
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

    # download DVAE files if needed
    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print(" > Downloading DVAE files!")
        ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)


    # Download XTTS v2.0 checkpoint if needed
    TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
    XTTS_CONFIG_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"

    # XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
    #TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
    TOKENIZER_FILE = "/Users/nikitasaravanan/Downloads/XTTSv2-Finetuning-for-New-Languages/checkpoints/XTTS_v2.0_original_model_files/tokenizer_new/vocab.json"
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file
    XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CONFIG_LINK))  # config.json file

    # download XTTS v2.0 files if needed
    if not os.path.isfile(TOKENIZER_FILE):
        print(" > Downloading XTTS v2.0 tokenizer!")
        ModelManager._download_model_files(
            [TOKENIZER_FILE_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )
    if not os.path.isfile(XTTS_CHECKPOINT):
        print(" > Downloading XTTS v2.0 checkpoint!")
        ModelManager._download_model_files(
            [XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )
    if not os.path.isfile(XTTS_CONFIG_FILE):
        print(" > Downloading XTTS v2.0 config!")
        ModelManager._download_model_files(
            [XTTS_CONFIG_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )

    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=11025,  # 0.5 secs
        debug_loading_failures=False,
        max_wav_length=max_audio_length,  # ~11.6 seconds
        max_text_length=max_text_length,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # define audio config
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # training parameters config

    config = GPTTrainerConfig()

    config.load_json(XTTS_CONFIG_FILE)

    config.epochs = num_epochs
    config.output_path = OUT_PATH
    config.model_args = model_args
    config.run_name = RUN_NAME
    config.project_name = PROJECT_NAME
    config.run_description = """
        GPT XTTS training
        """
    config.dashboard_logger = DASHBOARD_LOGGER
    config.logger_uri = LOGGER_URI
    config.audio = audio_config
    config.batch_size = BATCH_SIZE
    config.num_loader_workers = 8
    config.eval_split_max_size = 256
    config.print_step = 50
    config.plot_step = 100
    config.log_model_step = 100
    config.save_step = save_step
    config.save_n_checkpoints = 1
    config.save_checkpoints = True
    config.print_eval = False
    config.optimizer = "AdamW"
    config.optimizer_wd_only_on_weights = OPTIMIZER_WD_ONLY_ON_WEIGHTS
    config.optimizer_params = {"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": weight_decay}
    config.lr = lr
    config.lr_scheduler = "MultiStepLR"
    config.lr_scheduler_params = {"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1}
    config.test_sentences = []

    # init the model from config
    model = GPTTrainer.init_from_config(config)
    model.to(device)
    print_memory_usage("After model.to(device)")

    # load training samples
    # === Inspect datasets using load_tts ===
    print("\n>>> Inspecting datasets using load_tts")
    
    print("loading samples...")
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    print(f"Loaded {len(train_samples)} train samples, {len(eval_samples)} eval samples")
    print_memory_usage("After loading train/eval samples")

    def summarize(samples, name):
        text_lengths = [len(s["text"]) for s in samples if "text" in s]
        audio_durations = [s.get("duration", 0) for s in samples]
        print(f"--- {name} ---")
        print(f"Samples: {len(samples)}")
        if text_lengths:
            print(f"Avg text length: {sum(text_lengths)/len(text_lengths):.2f} chars")
        if audio_durations:
            print(f"Avg audio duration: {sum(audio_durations)/len(audio_durations):.2f} sec")

    summarize(train_samples, "Train")
    summarize(eval_samples, "Eval")

    if gpt_checkpoint is not None:
        print(f">> Loading GPT model weights from: {gpt_checkpoint}")
        checkpoint = torch.load(gpt_checkpoint, map_location=device)
        model.load_state_dict(checkpoint, strict=False)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=gpt_checkpoint,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=os.path.join(output_path, "run", "training"),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # Start the resource monitoring thread to print usage every 50 seconds
    monitor_thread = threading.Thread(target=monitor_resources, args=(2,), daemon=True)
    monitor_thread.start()

    trainer.fit()

    # get the longest text audio file to use as speaker reference
    samples_len = [len(item["text"].split(" ")) for item in train_samples]
    longest_text_idx =  samples_len.index(max(samples_len))
    speaker_ref = train_samples[longest_text_idx]["audio_file"]

    trainer_out_path = trainer.output_path

    # deallocate VRAM and RAM
    del model, trainer, train_samples, eval_samples
    gc.collect()

    return trainer_out_path

if __name__ == "__main__":
    parser = create_xtts_trainer_parser()
    args = parser.parse_args()

    trainer_out_path = train_gpt(
        metadatas=args.metadatas,
        output_path=args.output_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_acumm=args.grad_acumm,
        weight_decay=args.weight_decay,
        lr=args.lr,
        max_text_length=args.max_text_length,
        max_audio_length=args.max_audio_length,
        save_step=args.save_step,
        gpt_checkpoint = args.gpt_checkpoint,
        start_epoch = args.start_epoch
        # optionally pass other args
    )

    print(f"Checkpoint saved in dir: {trainer_out_path}")