import sys
import os
import torch
import time
import speech_recognition as sr
import sounddevice as sd
import numpy as np
from utils.tools import prepare_text
import ollama
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

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

# Load TTS models at startup and keep them in memory
glados = torch.jit.load('models/glados.pt')
vocoder = torch.jit.load('models/vocoder-gpu.pt', map_location=device)

# Create a thread-safe queue for audio playback
audio_queue = queue.Queue()

# Initialize speech recognizer globally
recognizer = sr.Recognizer()
# Adjust these parameters for better recognition
recognizer.energy_threshold = 300  # Increase if in noisy environment
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 0.8  # Reduce for faster response

def glados_tts(text):
    """Optimized TTS generation"""
    x = prepare_text(text).to('cpu')
    with torch.no_grad():
        old_time = time.time()
        
        # Move operations to appropriate devices
        tts_output = glados.generate_jit(x)
        mel = tts_output['mel_post'].to(device)
        
        # Batch process audio if possible
        audio = vocoder(mel)
        audio = audio.squeeze() * 32768.0
        
        print(f"\033[1;94mINFO:\033[;97m Audio generated in {round((time.time() - old_time) * 1000)} ms.")
        return audio.cpu().numpy().astype('int16')

def audio_player_thread():
    """Separate thread for audio playback"""
    while True:
        audio_data = audio_queue.get()
        if audio_data is None:  # Shutdown signal
            break
        sd.play(audio_data, samplerate=22050)
        sd.wait()
        audio_queue.task_done()

def recognize_speech():
    """Optimized speech recognition"""
    with sr.Microphone() as source:
        print("\033[1;94mINFO:\033[;97m Listening...")
        # Only adjust for ambient noise occasionally
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            print(f"\033[1;94mINFO:\033[;97m You said: {text}")
            return text
        except (sr.UnknownValueError, sr.RequestError) as e:
            print(f"\033[1;91mERROR:\033[;97m {str(e)}")
            return None

def get_llm_response(prompt):
    """Optimized LLM response generation"""
    try:
        # Add system context to improve response quality and speed
        context = "You are GLaDOS, a witty AI. Keep responses concise and entertaining with a hint of sarcasm in your answer."
        full_prompt = f"{context}\nUser: {prompt}\nGLaDOS:"
        
        response = ollama.generate(
            model='llama3.2:3b',
            prompt=full_prompt,
            stream=False,
            options={
                'num_predict': 100,  # Limit response length
                'temperature': 0.7,  # Adjust for better balance of creativity/speed
                'top_p': 0.9,
            }
        )
        return response['response']
    except Exception as e:
        print(f"\033[1;91mERROR:\033[;97m {e}")
        return None

def process_response(user_input):
    """Process LLM response and TTS in parallel"""
    llm_output = get_llm_response(user_input)
    if llm_output:
        print(f"GLaDOS: {llm_output}")
        audio_data = glados_tts(llm_output)
        audio_queue.put(audio_data)

def main():
    print("\033[1;94mINFO:\033[;97m GLaDOS TTS System - Ready to chat! (Press Ctrl+C to quit)")
    
    # Start audio player thread
    audio_thread = threading.Thread(target=audio_player_thread, daemon=True)
    audio_thread.start()
    
    # Create thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            user_input = recognize_speech()
            if not user_input:
                continue
                
            if user_input.lower() == "exit":
                print("Exiting...")
                audio_queue.put(None)  # Signal audio thread to stop
                break
                
            # Process response in separate thread
            executor.submit(process_response, user_input)

if __name__ == "__main__":
    main()
