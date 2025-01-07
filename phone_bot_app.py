import json
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

class Config:
    def __init__(self, config_path='config.json'):
        try:
            with open(config_path, 'r') as file:
                self.config = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error parsing the configuration file: {e}")
            exit(1)
        except FileNotFoundError:
            print(f"Configuration file {config_path} not found.")
            exit(1)

    def get(self, *keys, default=None):
        data = self.config
        for key in keys:
            data = data.get(key, {})
        return data if data else default

class TokenLoader:
    def __init__(self, token_file):
        self.token = self.load_token(token_file)

    @staticmethod
    def load_token(token_file):
        try:
            with open(token_file, 'r') as file:
                return file.read().strip()
        except FileNotFoundError:
            print(f"Token file {token_file} not found.")
            exit(1)

class LlamaModel:
    def __init__(self, model_dir, model_file, n_gpu_layers, n_threads, use_mlock, use_cloud=False):
        self.use_cloud = use_cloud
        self.ionos_token = None  # Will be set later if needed
        if not self.use_cloud:
            self.model_path = f"{model_dir}/{model_file}"
            self.n_gpu_layers = n_gpu_layers
            self.n_threads = n_threads
            self.use_mlock = use_mlock
            self.model = self.load_model()
        else:
            self.model = None  # No local model loaded when using cloud
        print("LLM initialized.")

    def load_model(self):
        try:
            llm = Llama(
                model_path=self.model_path,
                device_map="balanced",
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                use_mlock=self.use_mlock
            )
            print("Llama model loaded.")
            return llm
        except Exception as e:
            print(f"Failed to load Llama model: {e}")
            exit(1)

    def generate_response(self, prompt, conversation_history, system_role, use_cloud, ionos_token, piper_url):
        if conversation_history is None:
            conversation_history = [{"role": "system", "content": system_role}]
        else:
            if not any(msg["role"] == "system" and msg["content"] == system_role for msg in conversation_history):
                conversation_history.insert(0, {"role": "system", "content": system_role})

        conversation_history.append({"role": "user", "content": prompt})

        if use_cloud:
            cloud_model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
            endpoint = "https://openai.inference.de-txl.ionos.com/v1/chat/completions"

            headers = {
                "Authorization": f"Bearer {ionos_token}",
                "Content-Type": "application/json"
            }
            body = {
                "model": cloud_model_name,
                "messages": conversation_history,
            }
            try:
                response = requests.post(endpoint, json=body, headers=headers).json()
                response_content = response['choices'][0]['message']['content']
            except Exception as e:
                print(f"Error during cloud request: {e}")
                return "Entschuldigung, ich konnte keine Antwort generieren.", conversation_history
        else:
            try:
                response = self.model.create_chat_completion(
                    messages=conversation_history,
                    max_tokens=100
                )
                response_content = response['choices'][0]['message']['content']
            except Exception as e:
                print(f"Error during local model inference: {e}")
                return "Entschuldigung, ich konnte keine Antwort generieren.", conversation_history

        conversation_history.append({"role": "assistant", "content": response_content})

        return response_content, conversation_history

class WhisperModel:
    def __init__(self, model_name, device):
        try:
            self.model = whisper.load_model(model_name, device=device)
            print("Whisper model loaded.")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            exit(1)

    def transcribe(self, audio_buffer, rate, language='de'):
        try:
            audio_array = np.frombuffer(audio_buffer, np.int16).astype(np.float32) / 32768.0
            result = self.model.transcribe(audio_array, fp16=False, language=language)
            return result.get("text", "").strip()
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""

class AudioHandler:
    def __init__(self, audio_config):
        try:
            self.format = getattr(pyaudio, audio_config['format'])
        except AttributeError:
            print(f"Invalid audio format: {audio_config['format']}")
            exit(1)
        self.channels = audio_config['channels']
        self.rate = audio_config['rate']
        self.chunk = int(self.rate * audio_config['frame_duration_ms'] / 1000)
        self.silence_threshold = audio_config['silence_threshold_sec']
        self.audio = pyaudio.PyAudio()
        self.stream = self.configure_stream()

    def configure_stream(self):
        try:
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            return stream
        except Exception as e:
            print(f"Failed to open audio stream: {e}")
            exit(1)

    def read_chunk(self):
        try:
            return self.stream.read(self.chunk, exception_on_overflow=False)
        except Exception as e:
            print(f"Error reading audio chunk: {e}")
            return b""

    def stop(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            print("Audio stream terminated.")
        except Exception as e:
            print(f"Error stopping audio stream: {e}")

    def play_audio(self, buffer):
        try:
            wav_stream = io.BytesIO()
            with wave.open(wav_stream, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.audio.get_sample_size(self.format))
                wav_file.setframerate(self.rate)
                wav_file.writeframes(buffer)
            wav_stream.seek(0)
            # Optionally, implement playback of the recorded buffer
        except Exception as e:
            print(f"Error during audio playback: {e}")

    def play_wave_bytes(self, audio_data):
        try:
            response_audio = io.BytesIO(audio_data)
            response_wave = wave.open(response_audio, 'rb')
            response_wave_obj = sa.WaveObject.from_wave_read(response_wave)
            response_play_obj = response_wave_obj.play()
            response_play_obj.wait_done()
        except Exception as e:
            print(f"Error playing response audio: {e}")

class SpeechSynthesizer:
    def __init__(self, piper_url):
        self.url = piper_url

    def synthesize(self, text):
        try:
            payload = {'text': text}
            response = requests.get(self.url, params=payload)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            print(f"Error during speech synthesis: {e}")
            return b""

class ConversationManager:
    def __init__(self, system_role):
        self.system_role = system_role
        self.history = []

class Application:
    def __init__(self, config_path='config.json'):
        self.config = Config(config_path)
        self.token_loader = TokenLoader(self.config.get('ionos_token_file'))
        self.use_cloud = self.config.get('use_cloud', default=False)
        llama_conf = self.config.get('llama_model')
        self.llama = LlamaModel(
            model_dir=llama_conf['directory'],
            model_file=llama_conf['file'],
            n_gpu_layers=llama_conf['n_gpu_layers'],
            n_threads=llama_conf['n_threads'],
            use_mlock=llama_conf['use_mlock'],
            use_cloud=self.use_cloud
        )
        self.llama.ionos_token = self.token_loader.token if self.use_cloud else None

        whisper_conf = self.config.get('whisper_model')
        device = torch.device(whisper_conf['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}\n")
        self.whisper = WhisperModel(
            model_name=whisper_conf['name'],
            device=device
        )
        self.vad = webrtcvad.Vad(3)
        audio_conf = self.config.get('audio')
        self.audio_handler = AudioHandler(audio_conf)
        piper_conf = self.config.get('piper')
        self.synthesizer = SpeechSynthesizer(piper_conf['url'])
        self.conversation = ConversationManager(self.config.get('system_role'))
        self.initial_message = self.config.get('initial_message')
        print(f'Jan: {self.initial_message}')
        self.silence_threshold = audio_conf['silence_threshold_sec']
        self.play_initial_message()

    def play_initial_message(self):
        audio_data = self.synthesizer.synthesize(self.initial_message)
        if audio_data:
            self.audio_handler.play_wave_bytes(audio_data)
        else:
            print("Failed to synthesize initial message.")

    def run(self):
        buffer = b""
        silence_start = None
        is_processing = False

        print("Listening and transcribing...")

        role_file = 'prompt_ionos.txt' if self.use_cloud else 'prompt.txt'
        try:
            with open(role_file, 'r') as file:
                system_role_content = file.read()
            self.conversation.history = [{"role": "system", "content": system_role_content}]
        except FileNotFoundError:
            print(f"Role file {role_file} not found. Using default system role.")
            self.conversation.history = [{"role": "system", "content": self.conversation.system_role}]

        try:
            while True:
                if is_processing:
                    # Stop the audio stream during processing
                    self.audio_handler.stream.stop_stream()

                    # Perform processing tasks
                    if buffer:
                        # Optionally, play the recorded buffer
                        # self.audio_handler.play_audio(buffer)
                        transcribed_text = self.whisper.transcribe(buffer, self.audio_handler.rate)
                        if transcribed_text:
                            print(f'User: {transcribed_text}')
                            llm_response, self.conversation.history = self.llama.generate_response(
                                prompt=transcribed_text,
                                conversation_history=self.conversation.history,
                                system_role=self.conversation.system_role,
                                use_cloud=self.use_cloud,
                                ionos_token=self.llama.ionos_token,
                                piper_url=self.config.get('piper', 'url')
                            )
                            print(f'Jan: {llm_response}')
                            audio_data = self.synthesizer.synthesize(llm_response)
                            if audio_data:
                                self.audio_handler.play_wave_bytes(audio_data)
                            else:
                                print("Failed to synthesize response audio.")
                        else:
                            print("No text transcribed, skipping.")

                    is_processing = False  # Reset processing flag
                    buffer = b""
                    silence_start = None

                    # Restart the audio stream after processing
                    self.audio_handler.stream.start_stream()
                    continue

                # Audio capture and VAD logic
                audio_chunk = self.audio_handler.read_chunk()
                if not audio_chunk:
                    continue

                is_speech = self.vad.is_speech(audio_chunk, self.audio_handler.rate)

                if is_speech:
                    buffer += audio_chunk
                    silence_start = None
                else:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > self.silence_threshold:
                        is_processing = True  # Set processing flag

        except KeyboardInterrupt:
            print("Stopping...")
            self.audio_handler.stop()

if __name__ == "__main__":
    app = Application('config.json')
    app.run()
