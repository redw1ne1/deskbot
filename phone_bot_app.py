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
    """
    Handles the loading and retrieval of configuration parameters from a JSON file.
    """
    def __init__(self, config_path='config.json'):
        """
        Initializes the Config object by loading the JSON configuration file.

        Args:
            config_path (str): Path to the configuration JSON file. Defaults to 'config.json'.
        """
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
        """
        Retrieves a nested configuration value based on the provided keys.

        Args:
            *keys: Variable length argument list representing the hierarchy of keys.
            default: Default value to return if the specified keys are not found.

        Returns:
            The value corresponding to the provided keys or the default value.
        """
        data = self.config
        for key in keys:
            data = data.get(key, {})
        return data if data else default

class TokenLoader:
    """
    Manages the loading of authentication tokens from a file.
    """
    def __init__(self, token_file):
        """
        Initializes the TokenLoader by loading the token from the specified file.

        Args:
            token_file (str): Path to the token file.
        """
        self.token = self.load_token(token_file)

    @staticmethod
    def load_token(token_file):
        """
        Reads and returns the token from the given file.

        Args:
            token_file (str): Path to the token file.

        Returns:
            str: The loaded token.

        Raises:
            SystemExit: If the token file is not found.
        """
        try:
            with open(token_file, 'r') as file:
                return file.read().strip()
        except FileNotFoundError:
            print(f"Token file {token_file} not found.")
            exit(1)

class LlamaModel:
    """
    Handles loading and interacting with the Llama language model, supporting both local and cloud-based operations.
    """
    def __init__(self, model_dir, model_file, n_gpu_layers, n_threads, use_mlock, use_cloud=False):
        """
        Initializes the LlamaModel with the provided parameters.

        Args:
            model_dir (str): Directory where the Llama model is stored.
            model_file (str): Filename of the Llama model.
            n_gpu_layers (int): Number of layers to offload to the GPU.
            n_threads (int): Number of CPU threads for token generation.
            use_mlock (bool): Whether to lock the model in memory.
            use_cloud (bool, optional): Flag to determine if cloud-based LLM should be used. Defaults to False.
        """
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
        """
        Loads the Llama model from the specified path with the given configurations.

        Returns:
            Llama: An instance of the loaded Llama model.

        Raises:
            SystemExit: If the model fails to load.
        """
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

    def generate_response(self, prompt, conversation_history, system_role, use_cloud, ionos_token):
        """
        Generates a response from the Llama model based on the provided prompt and conversation history.

        Args:
            prompt (str): User input to generate a response for.
            conversation_history (list): List of previous conversation messages.
            system_role (str): Role description for the system in the conversation.
            use_cloud (bool): Flag indicating whether to use the cloud-based LLM.
            ionos_token (str): Authentication token for cloud-based requests.

        Returns:
            tuple: The generated response content and the updated conversation history.
        """
        if conversation_history is None:
            conversation_history = [{"role": "system", "content": system_role}]
        else:
            # Ensure the system role is the first entry in the conversation history
            if not any(msg["role"] == "system" and msg["content"] == system_role for msg in conversation_history):
                conversation_history.insert(0, {"role": "system", "content": system_role})

        # Append the user's input to the conversation history
        conversation_history.append({"role": "user", "content": prompt})

        if use_cloud:
            # Configuration for cloud-based LLM
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
                # Send POST request to the cloud endpoint
                response = requests.post(endpoint, json=body, headers=headers).json()
                response_content = response['choices'][0]['message']['content']
            except Exception as e:
                print(f"Error during cloud request: {e}")
                # Return a default apology message if cloud request fails
                return "Entschuldigung, ich konnte keine Antwort generieren.", conversation_history
        else:
            try:
                # Generate response using the local Llama model
                response = self.model.create_chat_completion(
                    messages=conversation_history,
                    max_tokens=100
                )
                response_content = response['choices'][0]['message']['content']
            except Exception as e:
                print(f"Error during local model inference: {e}")
                # Return a default apology message if local inference fails
                return "Entschuldigung, ich konnte keine Antwort generieren.", conversation_history

        # Append the assistant's response to the conversation history
        conversation_history.append({"role": "assistant", "content": response_content})

        return response_content, conversation_history

class WhisperModel:
    """
    Manages the Whisper model for transcribing audio input into text.
    """
    def __init__(self, model_name, device):
        """
        Initializes the WhisperModel by loading the specified Whisper model.

        Args:
            model_name (str): Name of the Whisper model to load.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        try:
            self.model = whisper.load_model(model_name, device=device)
            print("Whisper model loaded.")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            exit(1)

    def transcribe(self, audio_buffer, rate, language='de'):
        """
        Transcribes the provided audio buffer into text using the Whisper model.

        Args:
            audio_buffer (bytes): Audio data to transcribe.
            rate (int): Sampling rate of the audio.
            language (str, optional): Language of the audio. Defaults to 'de' (German).

        Returns:
            str: The transcribed text.

        Raises:
            Exception: If transcription fails.
        """
        try:
            # Convert audio buffer to a numpy array with normalized float values
            audio_array = np.frombuffer(audio_buffer, np.int16).astype(np.float32) / 32768.0
            # Perform transcription
            result = self.model.transcribe(audio_array, fp16=False, language=language)
            return result.get("text", "").strip()
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""

class AudioHandler:
    """
    Handles audio input and output operations, including configuring audio streams and playing audio.
    """
    def __init__(self, audio_config):
        """
        Initializes the AudioHandler with the specified audio configuration.

        Args:
            audio_config (dict): Dictionary containing audio configuration parameters.
        """
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
        """
        Configures and opens the audio input stream based on the initialized parameters.

        Returns:
            pyaudio.Stream: The configured audio input stream.

        Raises:
            Exception: If the audio stream fails to open.
        """
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
        """
        Reads a chunk of audio data from the input stream.

        Returns:
            bytes: The read audio data chunk.

        Raises:
            Exception: If reading the audio chunk fails.
        """
        try:
            return self.stream.read(self.chunk, exception_on_overflow=False)
        except Exception as e:
            print(f"Error reading audio chunk: {e}")
            return b""

    def stop(self):
        """
        Stops and closes the audio stream, and terminates the PyAudio instance.
        """
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            print("Audio stream terminated.")
        except Exception as e:
            print(f"Error stopping audio stream: {e}")

    def play_audio(self, buffer):
        """
        Prepares the audio buffer for playback. Playback implementation is optional and can be added as needed.

        Args:
            buffer (bytes): Audio data buffer to be played.
        """
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
        """
        Plays the provided WAV audio data.

        Args:
            audio_data (bytes): WAV audio data to play.
        """
        try:
            response_audio = io.BytesIO(audio_data)
            response_wave = wave.open(response_audio, 'rb')
            response_wave_obj = sa.WaveObject.from_wave_read(response_wave)
            response_play_obj = response_wave_obj.play()
            response_play_obj.wait_done()
        except Exception as e:
            print(f"Error playing response audio: {e}")

class SpeechSynthesizer:
    """
    Manages the synthesis of speech from text using an external TTS server.
    """
    def __init__(self, piper_url):
        """
        Initializes the SpeechSynthesizer with the specified Piper TTS server URL.

        Args:
            piper_url (str): URL of the Piper TTS server.
        """
        self.url = piper_url

    def synthesize(self, text):
        """
        Sends text to the Piper TTS server and retrieves the synthesized speech audio.

        Args:
            text (str): The text to synthesize into speech.

        Returns:
            bytes: The synthesized audio content.

        Raises:
            requests.RequestException: If the TTS request fails.
        """
        try:
            payload = {'text': text}
            response = requests.get(self.url, params=payload)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            print(f"Error during speech synthesis: {e}")
            return b""

class ConversationManager:
    """
    Maintains and manages the conversation history between the user and the assistant.
    """
    def __init__(self, system_role):
        """
        Initializes the ConversationManager with a system role.

        Args:
            system_role (str): Description of the system's role in the conversation.
        """
        self.system_role = system_role
        self.history = []

class Application:
    """
    Integrates all components and manages the main application loop for processing audio input and generating responses.
    """
    def __init__(self, config_path='config.json'):
        """
        Initializes the Application by setting up all necessary components based on the configuration.

        Args:
            config_path (str, optional): Path to the configuration JSON file. Defaults to 'config.json'.
        """
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
        self.vad = webrtcvad.Vad(3)  # Initialize Voice Activity Detector with aggressiveness mode 3
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
        """
        Synthesizes and plays the initial welcome message to the user.
        """
        audio_data = self.synthesizer.synthesize(self.initial_message)
        if audio_data:
            self.audio_handler.play_wave_bytes(audio_data)
        else:
            print("Failed to synthesize initial message.")

    def run(self):
        """
        Starts the main application loop, handling audio input, transcription, response generation, and audio output.
        """
        buffer = b""  # Buffer to accumulate audio data
        silence_start = None  # Timestamp when silence is detected
        is_processing = False  # Flag indicating whether the application is processing the buffered audio

        print("Listening and transcribing...")

        # Determine the role file based on the use_cloud flag
        role_file = 'prompt_ionos.txt' if self.use_cloud else 'prompt.txt'
        print(role_file)
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
                    # Stop the audio stream during processing to prevent capturing more audio
                    self.audio_handler.stream.stop_stream()

                    # Perform processing tasks on the accumulated audio buffer
                    if buffer:
                        # Optionally, play the recorded buffer
                        # self.audio_handler.play_audio(buffer)
                        transcribed_text = self.whisper.transcribe(buffer, self.audio_handler.rate)
                        if transcribed_text:
                            print(f'User: {transcribed_text}')
                            # Generate response using the Llama model (cloud or local)
                            llm_response, self.conversation.history = self.llama.generate_response(
                                prompt=transcribed_text,
                                conversation_history=self.conversation.history,
                                system_role=self.conversation.system_role,
                                use_cloud=self.use_cloud,
                                ionos_token=self.llama.ionos_token
                                #piper_url=self.config.get('piper', 'url')
                            )
                            print(f'Jan: {llm_response}')
                            # Synthesize the generated response into speech
                            audio_data = self.synthesizer.synthesize(llm_response)
                            if audio_data:
                                self.audio_handler.play_wave_bytes(audio_data)
                            else:
                                print("Failed to synthesize response audio.")
                        else:
                            print("No text transcribed, skipping.")

                    # Reset processing flags and buffer
                    is_processing = False
                    buffer = b""
                    silence_start = None

                    # Restart the audio stream to continue listening
                    self.audio_handler.stream.start_stream()
                    continue

                # Capture a chunk of audio data
                audio_chunk = self.audio_handler.read_chunk()
                if not audio_chunk:
                    continue

                # Detect if the captured chunk contains speech
                is_speech = self.vad.is_speech(audio_chunk, self.audio_handler.rate)

                if is_speech:
                    buffer += audio_chunk  # Accumulate speech audio
                    silence_start = None  # Reset silence timer
                else:
                    if silence_start is None:
                        silence_start = time.time()  # Mark the start of silence
                    elif time.time() - silence_start > self.silence_threshold:
                        is_processing = True  # Trigger processing if silence exceeds threshold

        except KeyboardInterrupt:
            # Gracefully handle termination via keyboard interrupt
            print("Stopping...")
            self.audio_handler.stop()

if __name__ == "__main__":
    # Instantiate and run the application
    app = Application('config.json')
    app.run()
