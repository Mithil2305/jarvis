
import torch
import soundfile as sf
import sounddevice as sd

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

MODEL_ID = "ai4bharat/indic-parler-tts"

print("Loading Indic Parler-TTS on CPU...", flush=True)

model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_ID
).to("cpu")
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)

text = (
    "Hello! I am Jarvis, your personal assistant. "
    "How can I help you today?"
)

description = (
    "A natural, conversational Indian English speaker "
    "speaks clearly at a moderate pace with a warm, "
    "friendly tone. The audio is clear and close, "
    "with no background noise."
)

print("Preparing text...", flush=True)

description_inputs = description_tokenizer(
    description,
    return_tensors="pt",
)

text_inputs = tokenizer(
    text,
    return_tensors="pt",
)

print("Generating speech. Please wait...", flush=True)

with torch.inference_mode():
    output = model.generate(
        input_ids=description_inputs.input_ids,
        attention_mask=description_inputs.attention_mask,
        prompt_input_ids=text_inputs.input_ids,
        prompt_attention_mask=text_inputs.attention_mask,
    )

audio = output.cpu().numpy().squeeze()

if audio.size == 0:
    raise RuntimeError("The model generated empty audio.")

output_file = "indic_jarvis_test.wav"

sf.write(
    output_file,
    audio,
    model.config.sampling_rate,
)

print(f"Audio saved to {output_file}", flush=True)
print("Playing generated speech...", flush=True)

sd.play(
    audio,
    samplerate=model.config.sampling_rate,
    blocking=True,
)

print("Audio playback completed.", flush=True)
