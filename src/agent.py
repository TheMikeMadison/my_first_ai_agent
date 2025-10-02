"""
My First AI Agent
A simple conversational agent using LangChain with local models.

This agent can:
- LangChain framework usage
- Local LLM integration with Ollama
- Memory and conversation chains
- Agent reasoning patterns
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# LangChain imports
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

# Load environment variables from .env file
load_dotenv()

class LangChainAgent():
    """
    A local AI agent powered by LangChain and Ollama

    Features:
    - Uses LangChain's conversation chains
    - Local LLM via Ollama (no API keys!)
    - Built-in memory management
    - Customizable prompts and personality
    """

    def __init__(self, name: str = None, model_name: str = "llama3.2:latest"):
        """
        Initialize the LangChain agent

        Args:
            name (str): The agent's name (defaults to environment variable)
            model_name (str): Ollama model to use
        """

        # Set up the agent's identity
        self.name = name or os.getenv("AGENT_NAME", "LangChainAgent")
        self.version = os.getenv("AGENT_VERSION", "1.1.0")
        # self.version = model_name
        self.model_name = model_name

        # Set up beautiful console output
        self.console = Console()

        # Initialize the local LLM
        self.console.print("[yellow]🔄 Connecting to local Ollama model...[/yellow]")

        try:
            self.llm = OllamaLLM(
                model=model_name,
                temperature=0.7,
                top_p=0.9,
            )

            # Test the connection
            test_response = self.llm.invoke("Hello")
            self.console.print("[green]✅ Successfully connected to Ollama![/green]")

        except Exception as e:
            self.console.print(f"[red]❌ Error connecting to Ollama: {e}[/red]")
            self.console.print("[yellow]Make sure Ollama is running: 'ollama serve'[/yellow]")
            raise
        
        # Set up conversation memory
        self.memory = ConversationBufferMemory(
            return_message=False,
            memory_key="history"
        )

        # Agent's personality and instructions
        self.prompt_template = PromptTemplate(
            input_variables=["history", "input"],
            template= f"""
            You are {self.name}, a helpful and friendly AI agent running locally.

            Your personality:
            - Helpful and informative
            - Curious and engaging  
            - Honest about your limitations
            - Encouraging and supportive
            - You love learning from humans

            Key principles:
            - Keep responses conversational and natural
            - Ask follow-up questions when appropriate
            - Admit when you don't know something
            - Show genuine interest in what the human shares
            - Remember you're running locally (no internet access)

            Previous conversation:
            {{history}}

            Human: {{input}}
            {self.name}:
            """
        )

        # Create the conversation chain
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt_template,
            verbose=False
        )

        self._display_startup_message()

    def _display_startup_message(self):
        """
        Display a welcome message when the agent starts.
        """
        startup_text = Text()
        startup_text.append(f"🤖 {self.name} v{self.version} (LangChain Powered) 🛡️\n", style="bold green")
        startup_text.append(f"Model: {self.model_name}\n", style="cyan")
        startup_text.append("Framework: LangChain + Ollama\n", style="blue")
        startup_text.append("\n🌟 Features:\n", style="bold yellow")
        startup_text.append("✅ Local LLM (no internet needed)\n", style="green")
        startup_text.append("✅ Conversation memory\n", style="green")
        startup_text.append("✅ No API keys or costs\n", style="green")
        startup_text.append("✅ Your data stays private\n", style="green")
        startup_text.append("Type 'quit' or 'exit' to end our conversation.", style="dim")

        self.console.print(Panel(startup_text, title="🦜⛓️ LangChain Agent Ready", border_style="green"))

    def perceive(self, user_input: str) -> str:
        """
        Perceive and process user input.

        This is the agent's sensory system - it receives and validates input.

        Args:
            user_input (str): Raw input from the user

        Returns:
            str: Processed and cleaned input
        """

        # Clean and validate the input
        processed_input = user_input.strip()

        if not processed_input:
            return ""

        return processed_input
    
    def think(self, user_input: str) -> str:
        """
        The agent's cognitive process - this is where the AI reasoning happens.

        Uses OpenAI's GPT model to process the input and generate a thoughtful response.

        Args:
            user_input (str): The user's message

        Returns:
            str: The agent's generated response
        """
        try:
            # LangChain handles the thinking process
            response = self.conversation.predict(input=user_input)

            response = response.strip()

            return response
        
        except Exception as e:
            self.console.print(f"I encountered an error while thinking: {str(e)} [/red]")
            return "I'm having trouble processing that right now. Could you try rephrasing?"
        
    def act(self, response: str):
        """
        The agent's action system - how it communicates back to the user.

        Args:
            response (str): The message to display to the user
        """

        # Create a beautiful response panel
        response_text = Text(response)
        self.console.print(Panel(
            response_text,
            title=f"🤖 {self.name}",
            border_style="blue",
            padding=(1,2)
        ))

    def chat(self):
        """
        Main conversation loop - this brings everything together

        This method orchestrates the full agent cycle:
        Perceive -> Think -> Act -> Repeat
        """
        try:
            while True:
                # Get user input
                self.console.print() # Spacing
                user_input = input("👤 You: ")

                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    goodbye_text = Text()
                    goodbye_text.append(f"Goodbye! It was great chatting with you! 👋", style="bold yellow")
                    self.console.print(Panel(goodbye_text, border_style="yellow"))
                    break
                
                # Agent perceives the input
                processed_input = self.perceive(user_input)

                if not processed_input:
                    self.console.print("[yellow]I didn't catch that. Could you say something?[/yellow]")
                    continue
                
                # Agent thinks and generates response
                self.console.print("[dim]🤔 Thinking with LangChain...[/dim]")
                response = self.think(processed_input)

                # Agent responds
                self.act(response)

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            self.console.print("\n[yellow]Chat interrupted. Goodbye! 👋[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]And unexpected error occurred: {str(e)}[/red]")

    def get_conversation_summary(self):
        """
        Get a summary of the conversation from memory.
        """
        if hasattr(self.memory, 'buffer'):
            return self.memory.buffer
        return "No conversation history yet."

# Function to create and run the agent
def main():
    """
    Main function to create and start the agent.
    """
    console = Console()

    console.print("[bold blue]🚀 Starting LangChain Agent...[/bold blue]")

    try:
        # Create the agent (you can change the model here)
        agent = LangChainAgent(
            name="LocalBot",
            model_name="llama3.2" # or "phi" for smaller/faster model
        )

        agent.chat()

    except Exception as e:
        console.print(f"[red]Failed to start agent: {str(e)}[/red]")
        console.print("[yellow]Make sure Ollama is running: 'ollama serve'[/yellow]")

if __name__ == "__main__":
    main()