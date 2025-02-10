import sys
import os
import re
import torch
import time
import base64
import sounddevice as sd
import numpy as np
from utils.tools import prepare_text
from scipy.io.wavfile import write
import wave

# Set up eSpeak environment
if sys.platform == "win32":
    os.system('color')
    if 'PHONEMIZER_ESPEAK_LIBRARY' not in os.environ or 'PHONEMIZER_ESPEAK_PATH' not in os.environ:
        espeak_base = r"C:\\Program Files\\eSpeak NG"
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = os.path.join(espeak_base, 'libespeak-ng.dll')
        os.environ['PHONEMIZER_ESPEAK_PATH'] = os.path.join(espeak_base, 'espeak-ng.exe')
        print("\033[1;94mINFO:\033[;97m Guessing eSpeak path..." + espeak_base, flush=True)

print("\033[1;94mINFO:\033[;97m Initializing TTS Engine...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\033[1;94mINFO:\033[;97m Torch ({torch.__version__}) will be using {device}", flush=True)

# Load TTS models
glados = torch.jit.load('models/glados.pt')
vocoder = torch.jit.load('models/vocoder-gpu.pt', map_location=device)

def glados_tts(text):
    x = prepare_text(text).to('cpu')
    with torch.no_grad():
        old_time = time.time()
        tts_output = glados.generate_jit(x)
        mel = tts_output['mel_post'].to(device)
        audio = vocoder(mel)
        print(f"\033[1;94mINFO:\033[;97m Audio generated in {round((time.time() - old_time) * 1000)} ms.")
        audio = audio.squeeze() * 32768.0
        return audio.cpu().numpy().astype('int16')

def play_audio(audio_data, sample_rate=22050):
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()

def main():
    while True:
        text = input("Enter text: ")
        if text.lower() == "exit":
            break
        audio_data = glados_tts(text)
        play_audio(audio_data)

if __name__ == "__main__":
    main()
