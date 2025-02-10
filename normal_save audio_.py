import sys
import os
import re
sys.path.insert(0, os.getcwd()+'/glados_tts')

import torch
from utils.tools import prepare_text
from scipy.io.wavfile import write
import time
import base64
import tkinter as tk
from tkinter import messagebox

if sys.platform == "win32":
    os.system('color')
    if 'PHONEMIZER_ESPEAK_LIBRARY' not in os.environ or 'PHONEMIZER_ESPEAK_PATH' not in os.environ:
        espeak_base = r"C:\Program Files\eSpeak NG"
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = os.path.join(espeak_base, 'libespeak-ng.dll')
        os.environ['PHONEMIZER_ESPEAK_PATH'] = os.path.join(espeak_base, 'espeak-ng.exe')
        print("\033[1;94mINFO:\033[;97m Guessing eSpeak path..." + espeak_base, flush=True)

print("\033[1;94mINFO:\033[;97m Initializing TTS Engine...")

# Select the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\033[1;94mINFO:\033[;97m Torch ({torch.__version__}) will be using {device}", flush=True)

# Load models
if __name__ == "__main__":
    glados = torch.jit.load('models/glados.pt')
    vocoder = torch.jit.load('models/vocoder-gpu.pt', map_location=device)
else:
    glados = torch.jit.load('glados_tts/models/glados.pt')
    vocoder = torch.jit.load('glados_tts/models/vocoder-gpu.pt', map_location=device)

# Prepare models in RAM
for i in range(4):
    init = glados.generate_jit(prepare_text(str(i)))
    init_mel = init['mel_post'].to(device)
    init_vo = vocoder(init_mel)

def sanitize_filename(text):
    return re.sub(r'[^a-zA-Z0-9_]', '_', text[:50])  # Limit filename length

def glados_tts(text):
    x = prepare_text(text).to('cpu')
    with torch.no_grad():
        old_time = time.time()
        tts_output = glados.generate_jit(x)
        mel = tts_output['mel_post'].to(device)
        audio = vocoder(mel)
        print(f"\033[1;94mINFO:\033[;97m Audio generated in {round((time.time() - old_time) * 1000)} ms.")
        audio = audio.squeeze() * 32768.0
        audio = audio.cpu().numpy().astype('int16')
        filename = f"audio/{sanitize_filename(text)}.wav"
        write(filename, 22050, audio)
        return base64.b64encode(open(filename, "rb").read())

def on_submit():
    text = text_entry.get("1.0", "end-1c").strip()
    if not text:
        messagebox.showwarning("Input Error", "Please enter some text.")
        return
    print(f"\033[1;94mINFO:\033[;97m Generating audio for: {text}")
    glados_tts(text)
    messagebox.showinfo("Success", "Audio generated successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("GLaDOS TTS")
    text_entry = tk.Text(root, height=10, width=50)
    text_entry.pack(pady=10)
    submit_button = tk.Button(root, text="Generate Audio", command=on_submit)
    submit_button.pack(pady=10)
    root.mainloop()
