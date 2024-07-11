import torch
from transformers import pipeline
import librosa #can handle audio files
import io
#from datasets import load_dataset

def convert_bytes_to_array(audio_bytes):
    audio_bytes = io.BytesIO(audio_bytes)
    audio, sample_rate = librosa.load(audio_bytes)
    print(sample_rate)
    return audio


def transcribe_audio(audio_bytes):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30, #max length 30 seconds
        device=device,
        )

    audio_array = convert_bytes_to_array(audio_bytes)
    prediction = pipe(audio_array, batch_size=8)["text"]

    return prediction


