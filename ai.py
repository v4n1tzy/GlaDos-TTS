import sys
import os
import torch
import time
import sounddevice as sd
import numpy as np
from utils.tools import prepare_text
import ollama  # For interacting with the Ollama LLM

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
    print("\033[1;94mINFO:\033[;97m GLaDOS TTS System - Ready to chat! (Type 'exit' to quit)")

    while True:
        # Get user input
        prompt = input("You: ")

        # Exit the loop if the user types 'exit'
        if prompt.lower() == "exit":
            print("Exiting...")
            break

        # Get the final response from the LLM
        llm_output = get_llm_response(prompt)
        if llm_output:
            print(f"GLaDOS: {llm_output}")

            # Generate and play audio using the GLaDOS TTS model
            audio_data = glados_tts(llm_output)
            play_audio(audio_data)

if __name__ == "__main__":
    main()