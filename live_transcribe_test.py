import webrtcvad
import whisper
import pyaudio
import numpy as np
import time
import torch
import requests
import simpleaudio as sa
import io
import wave
from llama_cpp import Llama

with open('ionos_token.txt', 'r') as file:
    ionos_token = file.read()

def load_model(model_path, n_gpu_layers=40, n_threads=12, use_mlock=True):
    """
    Loads the Llama model.

    Args:
        model_path (str): Path to the model file.
        n_gpu_layers (int): Number of layers to offload to GPU.
        n_threads (int): Number of CPU threads for token generation.
        use_mlock (bool): Whether to lock the model in memory.

    Returns:
        Llama: The loaded model instance.
    """
    llm = Llama(model_path=model_path, device_map="balanced", n_gpu_layers=n_gpu_layers)
    print("Model loaded")
    return llm

def generate_response(llm, prompt, conversation_history=None,
                      system_role="Du bist ein hilfreicher Assistent", use_cloud=False):
    """
    Generates a response from the model based on the given prompt, system role, and conversation history.

    Args:
        llm (Llama): The loaded Llama model.
        prompt (str): User input for the model.
        conversation_history (list): List of previous conversation messages.
        system_role (str): Role of the system in the conversation.
        use_cloud (Bool): True if you want to use cloud-based LLM

    Returns:
        tuple: The model's response and the updated conversation history.
    """
    if conversation_history is None:
        # Initialize with system role if no history exists
        conversation_history = [{"role": "system", "content": system_role}]
    else:
        # Ensure the system role is the first entry in the conversation history
        if not any(msg["role"] == "system" and msg["content"] == system_role for msg in conversation_history):
            conversation_history.insert(0, {"role": "system", "content": system_role})

        # Append the user's input to the conversation history
    conversation_history.append({"role": "user", "content": prompt})

    # Generate response using the model
    if use_cloud:
        #cloud_model_name = "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8"
        cloud_model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        endpoint = "https://openai.inference.de-txl.ionos.com/v1/chat/completions"

        header = {
            "Authorization": f"Bearer {ionos_token}",
            "Content-Type": "application/json"
        }
        body = {
            "model": cloud_model_name,
            "messages": conversation_history,
        }
        response = requests.post(endpoint, json=body, headers=header).json()
    else:
        response = llm.create_chat_completion(
            messages=conversation_history,
            max_tokens=100
        )
    response_content = response['choices'][0]['message']['content']

    # Append the assistant's response to the conversation history
    conversation_history.append({"role": "assistant", "content": response_content})

    return response_content, conversation_history

def configure_audio_stream(format, channels, rate, chunk):
    """
    Configures and opens the audio stream.

    Args:
        format: PyAudio format.
        channels (int): Number of audio channels.
        rate (int): Sampling rate.
        chunk (int): Chunk size in samples.

    Returns:
        PyAudio.Stream: The configured audio stream.
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)
    return audio, stream

def play_audio(buffer, audio, format, channels, rate):
    """
    Plays the recorded audio buffer.

    Args:
        buffer (bytes): Audio buffer.
        audio (pyaudio.PyAudio): PyAudio instance.
        format: Audio format.
        channels (int): Number of channels.
        rate (int): Sampling rate.
    """
    wav_stream = io.BytesIO()
    with wave.open(wav_stream, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(audio.get_sample_size(format))
        wav_file.setframerate(rate)
        wav_file.writeframes(buffer)
    wav_stream.seek(0)

def transcribe_audio(buffer, model, rate):
    """
    Transcribes audio using the Whisper model.

    Args:
        buffer (bytes): Audio buffer.
        model: Whisper model instance.
        rate (int): Sampling rate.

    Returns:
        str: Transcribed text.
    """
    audio_array = np.frombuffer(buffer, np.int16).astype(np.float32) / 32768.0
    result = model.transcribe(audio_array, fp16=False, language='de')
    return result.get("text", "").strip()

def synthesize_speech(text, url):
    """
    Sends text to the Piper TTS server and retrieves audio response.

    Args:
        text (str): Text to synthesize.
        url (str): Piper server URL.

    Returns:
        bytes: Audio response.
    """
    payload = {'text': text}
    response = requests.get(url, params=payload)
    return response.content

def play_response_audio(audio_data):
    """
    Plays the audio response from TTS.

    Args:
        audio_data (bytes): Audio data from TTS.
    """
    response_audio = io.BytesIO(audio_data)
    response_wave = wave.open(response_audio, 'rb')
    response_wave_obj = sa.WaveObject.from_wave_read(response_wave)
    response_play_obj = response_wave_obj.play()
    response_play_obj.wait_done()

def main_loop(llm, whisper_model, vad, audio_stream, audio, url, format, channels, rate, chunk, silence_threshold):
    """
    Main processing loop for voice-to-text-to-LLM-to-voice pipeline.

    Args:
        llm: Loaded Llama model.
        whisper_model: Loaded Whisper model.
        vad: WebRTC Voice Activity Detector instance.
        audio_stream: PyAudio audio stream.
        audio: PyAudio instance.
        url: Piper TTS server URL.
        format: Audio format.
        channels (int): Number of audio channels.
        rate (int): Sampling rate.
        chunk (int): Audio chunk size.
        silence_threshold (float): Silence threshold in seconds.
    """
    buffer = b""
    silence_start = None
    conversation_history = []  # Initialize conversation history

    print("Listening and transcribing...")

    if use_cloud:
        role_file = 'prompt_ionos.txt'
    else:
        role_file = 'prompt.txt'

    with open(role_file, 'r') as file:
        role = file.read()

    is_processing = False

    try:
        while True:
            if is_processing:
                # Stop the audio stream during processing
                audio_stream.stop_stream()

                # Perform processing tasks
                play_audio(buffer, audio, format, channels, rate)
                transcribed_text = transcribe_audio(buffer, whisper_model, rate)
                if transcribed_text:
                    print(f'User: {transcribed_text}')
                    llm_response, conversation_history = generate_response(
                        llm, transcribed_text, conversation_history, system_role=role, use_cloud=use_cloud
                    )
                    print(f'Jan: {llm_response}')
                    audio_data = synthesize_speech(llm_response, url)
                    play_response_audio(audio_data)
                else:
                    print("No text transcribed, skipping.")

                is_processing = False  # Reset processing flag
                buffer = b""
                silence_start = None

                # Restart the audio stream after processing
                audio_stream.start_stream()
                continue

            # Audio capture and VAD logic
            audio_chunk = audio_stream.read(chunk, exception_on_overflow=False)
            if len(audio_chunk) != chunk * 2:
                continue

            is_speech = vad.is_speech(audio_chunk, rate)

            if is_speech:
                buffer += audio_chunk
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > silence_threshold:
                    is_processing = True  # Set processing flag
    except KeyboardInterrupt:
        print("Stopping...")
        audio_stream.stop_stream()
        audio_stream.close()
        audio.terminate()


if __name__ == "__main__":
    llama_model_directory = "/home/radwan/Desktop/startUp_stuff/llama"
    llama_model_file = "Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf"
    #llama_model_file = "Llama-3.2-1B-Instruct-Q6_K_L.gguf"
    #llama_model_file = "Llama-3.2-3B-Instruct-Q6_K_L.gguf"
    llama_model_path = f"{llama_model_directory}/{llama_model_file}"

    use_cloud = True

    piper_url = "http://localhost:5000"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    if not use_cloud:
        llm = load_model(model_path=llama_model_path)
    else:
        llm = None

    whisper_model = whisper.load_model("turbo", device=device)
    vad = webrtcvad.Vad(3)

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    FRAME_DURATION = 30
    CHUNK = int(RATE * FRAME_DURATION / 1000)
    SILENCE_THRESHOLD = 1

    audio, audio_stream = configure_audio_stream(FORMAT, CHANNELS, RATE, CHUNK)

    initial_message = "Hallo und Willkommen. Hier ist Jan. Was kann ich für Sie tun?"
    audio_data = synthesize_speech(initial_message, piper_url)
    play_response_audio(audio_data)

    main_loop(llm, whisper_model, vad, audio_stream, audio, piper_url, FORMAT, CHANNELS, RATE, CHUNK, SILENCE_THRESHOLD)
