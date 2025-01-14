from llama_cpp import Llama
import requests
import json

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
    def __init__(self):
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

        self.config = Config()
        self.ionos_token = None  # Will be set later if needed
        self.use_cloud = self.config.get('use_cloud', default=False)
        self.conversation = ConversationManager(self.config.get('system_role'))

        if not self.use_cloud:
            llama_conf = self.config.get('llama_model')
            self.model_dir = llama_conf['directory']
            self.model_file = llama_conf['file']
            self.model_path = f"{model_dir}/{model_file}"
            self.n_gpu_layers = llama_conf['n_gpu_layers']
            self.n_threads = llama_conf['n_threads']
            self.use_mlock = llama_conf['use_mlock']
            self.model = self.load_model()
        else:
            llama_conf = self.config.get('llama_cloud_config')
            self.token_loader = TokenLoader(llama_conf['ionos_token_file'])
            self.ionos_token = self.token_loader.token
            self.model = llama_conf['cloud_model_name']
            self.endpoint = llama_conf['endpoint']

        role_file = llama_conf['role_file']
        try:
            with open(role_file, 'r') as file:
                self.system_role_content = file.read()
            self.conversation.history = [{"role": "system", "content": self.system_role_content}]
        except FileNotFoundError:
            print(f"Role file {role_file} not found. Using default system role.")
            self.conversation.history = [{"role": "system", "content": self.conversation.system_role}]

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

    def generate_response(self, prompt, conversation_history):
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
            conversation_history = self.conversation.history
        else:
            # Ensure the system role is the first entry in the conversation history
            if not any(msg["role"] == "system" and msg["content"] == self.system_role_content for msg in conversation_history):
                conversation_history.insert(0, {"role": "system", "content": self.system_role_content})

        # Append the user's input to the conversation history
        conversation_history.append({"role": "user", "content": prompt})

        if self.use_cloud:
            # Configuration for cloud-based LLM

            headers = {
                "Authorization": f"Bearer {self.ionos_token}",
                "Content-Type": "application/json"
            }
            body = {
                "model": self.model,
                "messages": conversation_history,
                "temperature": 0.5
            }
            try:
                # Send POST request to the cloud endpoint
                response = requests.post(self.endpoint, json=body, headers=headers).json()
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