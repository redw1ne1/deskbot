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


        llama_conf = self.config.get('llama_model')
        model_dir = llama_conf['directory']
        model_file = llama_conf['file']
        self.model_path = f"{model_dir}/{model_file}"
        self.n_gpu_layers = llama_conf['n_gpu_layers']
        self.n_ctx = llama_conf['n_ctx']
        self.n_threads = llama_conf['n_threads']
        self.use_mlock = llama_conf['use_mlock']

        # if not self.use_cloud:
        self.model_local = self.load_model()

        #else:
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
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                #n_threads=self.n_threads,
                #use_mlock=self.use_mlock
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

        agents = Agents(prompt, self.model_local, conversation_history)
        identify_purpose = agents.identify_purpose()
        print(identify_purpose)

        # The small talk branch - keep looping in small talk until the patient wants something else
        if identify_purpose == "small_talk":
            response_content, response_message = getattr(agents, identify_purpose)()
            print(response_content)
            conversation_history.append(response_message)
            return response_content, conversation_history



        #####################################################################################

        #                 Blocker to for LLM function calling

        #function_to_call = self.function_calling(prompt)
        """function_to_call = self.function_calling_local(prompt)
        if isinstance(function_to_call, str):
            function_to_call = json.loads(function_to_call)
        function_result = self.execute_function_call(function_to_call)
        print(function_result)
        conversation_history.append({"role": "tool", "content": function_result})
        print(conversation_history)"""
        #result = self.execute_function_call(function_to_call)


        #####################################################################################

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
                response = self.model_local.create_chat_completion(
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


    def function_calling_local(self, prompt):


        messages = [
            {"role": "system", "content": "du schlägst die passende Funktion vor. Wenn der erforderliche "
                                          "Eingabeparameter nicht angegeben ist, fordern Sie den Benutzer auf, ihn anzugeben. "
                                          "Auf keinen Fall machen Sie Annahmen."},
            {"role": "user", "content": prompt}
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_address",
                    "description": "Ruft die Adresse ab. Diese Funktion wird nur verwendet, wenn explizit nach einer Adresse gefragt wird.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_phonenumber",
                    "description": "Ruft die Telefonnummer ab. Diese Funktion wird nur verwendet, wenn explizit nach einer Telefonnummer gefragt wird.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_doctors",
                    "description": "Liefert Informationen über Ärzte basierend auf ihrer Spezialisierung. "
                                   "Diese Funktion wird nur verwendet, wenn eine Spezialisierung angegeben ist.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "specialization": {
                                "type": "string",
                                "description": "Die Spezialisierung des Arztes, nach der gesucht wird."
                            }
                        },
                        "required": ["specialization"]
                    },
                    "strict": True
                }
            }
        ]

        response = self.model_local.create_chat_completion(
            messages=messages,
            tools=tools
        )

        response_content = response['choices'][0]['message']['content']
        print(response_content)
        return response_content


    def execute_function_call(self, response_content):
        function_registry = {
            "get_address": self.get_address,
            "get_phonenumber": self.get_phonenumber,
            "get_doctors": self.get_doctors,
        }
        try:
            function_name = response_content.get("name")
            parameters = response_content.get("parameters", {})

            # Call the corresponding function
            if function_name in function_registry:
                function_to_call = function_registry[function_name]
                result = function_to_call(**parameters)  # Unpack parameters as arguments
                return result
            else:
                raise ValueError(f"Function '{function_name}' is not registered.")
        except Exception as e:
            return {"error": str(e)}


    def get_address(self):
        clinic_address = "blabla 13, Frankfurt"
        return clinic_address

    def get_phonenumber(self):
        clinic_number = "111 222 333"
        return clinic_number

    def get_doctors(self, specialization):
        doctors = ['Andreas Bauer', 'Andrea Bauerin', 'Andres Bilder']
        return doctors


    def function_calling(self, prompt):

        tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_info",
                        "description": "Sie geben informationen über Addresse und Telefonnummer",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "address": {
                                    "type": "boolean",
                                    "description": "ob die Addresse gefordert wird"
                                },
                                "phone_number": {
                                    "type": "boolean",
                                    "description": "ob die Telefonnummer gefordert wird"
                                }
                            },
                            "required": ["address", "phone_number"]
                        },
                    }
                }
            ]

        api_json = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "du schlägst die passende Funktion vor."},
                {"role": "user", "content": prompt}
                ],
            "tools": tools,
            "tool_choice": "required"
        }


        headers = {
            "Authorization": f"Bearer {self.ionos_token}",
            "Content-Type": "application/json"
        }
        """body = {
            "model": self.model,
            "messages": tools
        }"""
        # Send POST request to the cloud endpoint
        response = requests.post(self.endpoint, json=api_json, headers=headers).json()
        response_content = response['choices'][0]['message']['content']
        print("response: ", response)
        return response_content

class Agents:

    def __init__(self, prompt, model, conversation_history):
        """
        Description here.

        Args:
            None.
        """
        self.prompt = prompt
        self.model = model
        self.conversation_history = conversation_history


    def identify_purpose(self):

        system_role_definition = """Sie haben nur eine einzige Aufgabe: den Zweck der Anfrage zu ermitteln. 
        Wenn die Anfrage ausdrücklich nach der Adresse, der Telefonnummer oder den Ärzten der Praxis fragt, 
        schlagen Sie vor, mit get_info zu antworten. Wenn die Anfrage explizit nach Terminen, Symptomen, 
        Rezepten oder Krankmeldungen fragt, antworten Sie mit interactive. Wenn keine der vorherigen Anfragen explizit 
        erwähnt wird, nur dann schlagen Sie die Antwort small_talk vor."""

        messages = [
            {"role": "system", "content": system_role_definition},
            {"role": "user", "content": self.prompt}
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_info",
                    "description": "nur, wenn sie nach Praxisadresse, Telefonnummer oder Ärzten gefragt werden",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "interactive",
                    "description": "Wenn die Anfrage explizit nach Terminen, Symptomen, Rezepten oder Krankmeldungen fragt",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "small_talk",
                    "description": "wenn keine eindeutigen Angaben zu Praxisadresse, Telefonnummer, Ärzten, Terminen, "
                                   "Symptomen, Rezepten oder Krankmeldungen gemacht werden.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    },
                    "strict": True
                }
            }
        ]

        response = self.model.create_chat_completion(
            messages=messages,
            tools=tools,
            tool_choice="required"
        )
        response_content = response['choices'][0]['message']['content']
        response_content = json.loads(response_content)
        return response_content.get("name")

    def small_talk(self):

        system_role_definition = """Sie haben eine Aufgabe und nur eine Aufgabe: das Gespräch in einer respektvollen Art 
        (in Sie-Form) und Weise weiterzuführen. Wenn Sie den Namen des Anrufers kennen, sollten Sie ihn häufig verwenden. 
        Halten Sie Ihre Antwort kurz und fragen Sie, wie Sie dem Anrufer helfen können."""

        messages = [
            {"role": "system", "content": system_role_definition},
            {"role": "user", "content": self.prompt}
        ]

        response = self.model.create_chat_completion(
            messages=messages)

        response_message = response['choices'][0]['message']
        response_content = response_message['content']
        return response_content, response_message

    def get_info(self):

