import torch
import torchaudio
from tqdm import tqdm
from underthesea import sent_tokenize

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Device configuration
device = "mps" if torch.mps.is_available() else "cpu"

# Model paths
xtts_checkpoint = "/Users/nikitasaravanan/Downloads/XTTSv2-Finetuning-for-New-Languages/azure_checkpoints/GPT_XTTS_FT-July-09-2025_12+51AM-0000000/best_model_12085.pth"
xtts_config = "/Users/nikitasaravanan/Downloads/XTTSv2-Finetuning-for-New-Languages/azure_checkpoints/GPT_XTTS_FT-July-09-2025_12+51AM-0000000/config.json"
xtts_vocab = "/Users/nikitasaravanan/Downloads/XTTSv2-Finetuning-for-New-Languages/checkpoints/XTTS_v2.0_original_model_files/tokenizer_new/vocab.json"

# Load model
config = XttsConfig()
config.load_json(xtts_config)
XTTS_MODEL = Xtts.init_from_config(config)
XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
XTTS_MODEL.to(device)

print("Model loaded successfully!")

# Inference
tts_text = "|நீங்க பயன்படுத்தி நீங்களும் அதை வந்து பட்டியலிட்டு இருக்கீங்க"
speaker_audio_file = "/Users/nikitasaravanan/Downloads/XTTSv2-Finetuning-for-New-Languages/datasets-1_prev/wavs/taf_00008_00083252668.wav"
lang = "ta"

gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
    audio_path=speaker_audio_file,
    gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
    max_ref_length=XTTS_MODEL.config.max_ref_len,
    sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
)

tts_texts = sent_tokenize(tts_text)

wav_chunks = []
for text in tqdm(tts_texts):
    wav_chunk = XTTS_MODEL.inference(
        text=text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.1,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=10,
        top_p=0.3,
    )
    wav_chunks.append(torch.tensor(wav_chunk["wav"]))

out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()

# Play audio (for Jupyter Notebook)
#from IPython.display import Audio
#Audio(out_wav, rate=24000)

# Save the generated audio to a file
torchaudio.save("output_tamil_sg.wav", out_wav, 24000)
