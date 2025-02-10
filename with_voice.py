import sys
import os
import torch
import time
import speech_recognition as sr
import sounddevice as sd
import numpy as np
from utils.tools import prepare_text
import ollama  # For interacting with the Ollama LLM

# Set up eSpeak environment (if needed)
if sys.platform == "win32":
    os.system('color')
    if 'PHONEMIZER_ESPEAK_LIBRARY' not in os.environ or 'PHONEMIZER_ESPEAK_PATH' not in os.environ:
        espeak_base = r"C:\\Program Files\\eSpeak NG"
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = os.path.join(espeak_base, 'libespeak-ng.dll')
        os.environ['PHONEMIZER_ESPEAK_PATH'] = os.path.join(espeak_base, 'espeak-ng.exe')
        print("\033[1;94mINFO:\033[;97m Guessing eSpeak path..." + espeak_base, flush=True)

print("\033[1;94mINFO:\033[;97m Initializing TTS Engine...")

# Set up device (GPU or CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\033[1;94mINFO:\033[;97m Torch ({torch.__version__}) will be using {device}", flush=True)

# Load TTS models
glados = torch.jit.load('models/glados.pt')
vocoder = torch.jit.load('models/vocoder-gpu.pt', map_location=device)

def glados_tts(text):
    """
    Generate speech from text using the GLaDOS TTS model.
    :param text: The text to synthesize.
    :return: Audio data as a numpy array.
    """
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
    """
    Play audio data using sounddevice.
    :param audio_data: The audio data to play.
    :param sample_rate: The sample rate of the audio.
    """
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()

def recognize_speech():
    """
    Recognize speech from the microphone using Google Speech Recognition.
    :return: The recognized text, or None if recognition fails.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\033[1;94mINFO:\033[;97m Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print(f"\033[1;94mINFO:\033[;97m You said: {text}")
            return text
        except sr.UnknownValueError:
            print("\033[1;91mERROR:\033[;97m Could not understand audio.")
        except sr.RequestError:
            print("\033[1;91mERROR:\033[;97m Speech Recognition service unavailable.")
        return None

def get_llm_response(prompt):
    """
    Get the final response from the Ollama LLM.
    :param prompt: The user's input prompt.
    :return: The final response from the LLM.
    """
    try:
        # Generate the response using Llama3.2:3b
        response = ollama.generate(model='llama3.2:3b', prompt=prompt, stream=False)
        return response['response']
    except Exception as e:
        print(f"\033[1;91mERROR:\033[;97m {e}")
        return None

def main():
    print("\033[1;94mINFO:\033[;97m GLaDOS TTS System - Ready to chat! (Press Ctrl+C to quit)")

    while True:
        # Recognize speech from the user
        user_input = recognize_speech()
        if not user_input:
            continue

        # Exit the loop if the user says "exit"
        if user_input.lower() == "exit":
            print("Exiting...")
            break

        # Get the final response from the LLM
        llm_output = get_llm_response(user_input)
        if llm_output:
            print(f"GLaDOS: {llm_output}")

            # Generate and play audio using the GLaDOS TTS model
            audio_data = glados_tts(llm_output)
            play_audio(audio_data)

if __name__ == "__main__":
    main()