"""
My First AI Agent
A simple conversational agent that demonstrates basic agentic AI principles.

This agent can:
- Perceive user input
- Think using OpenAI's GPT model
- Respond intelligently
- Maintain conversation context
"""

import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Load environment variables from .env file
load_dotenv()

class SimpleAgent():
    """
    A basic AI agent that can have conversations with users.

    The agent follows the classic agent architecture:
    - Perceive: Receive user input
    - Think: Process using AI model
    - Act: Generate appropriate response
    """

    def __init__(self, name: str = None):
        """
        Initialize the agent with necessary components.

        Args:
            name (str): The agent's name (defaults to environment variable)
        """

        # Set up the agent's identity
        self.name = name or os.getenv("AGENT_NAME", "SimpleAgent")
        self.version = os.getenv("AGENT_VERSION", "1.0.0")

        # Initialize the OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Set up beautiful console output
        self.console = Console()

        # Memory: Store conversation history
        self.conversation_history: List[Dict[str, str]] = []

        # Agent's personality and instructions
        self.system_prompt = f"""
        You are {self.name}, a helpful and friendly AI agent.

        Your core principles:
        1. Be helpful and informative
        2. Ask clarifying questions when needed
        3. Admit when you don't know something
        4. Keep responses concise but thorough
        5. Show your reasoning process when helpful

        Remember: You are an AI agent learning to interact with humans effectively.
        """

        # Add a system prompt to conversation history
        self.conversation_history.append({
            "role": "system",
            "content": self.system_prompt
        })

        self._display_startup_message()

    def _display_startup_message(self):
        """
        Display a welcome message when the agent starts.
        """
        startup_text = Text()
        startup_text.append(f"🤖 {self.name} v{self.version} is now active!\n", style="bold green")
        startup_text.append("I'm an AI agent that can help you with various tasks.\n", style="white")
        startup_text.append("Type 'quit' or 'exit' to end our conversation.", style="dim")

        self.console.print(Panel(startup_text, title="Agent Status", border_style="green"))

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
        
        # Add to conversation memorty
        self.conversation_history.append({
            "role": "user",
            "content": processed_input
        })

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
            # Call OpenAI's API with our conversation history
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=self.conversation_history,
                max_tokens=500,
                temperature=0.7,    # Controls creativity (0.0 = very focused, 1.0 = very creative)
                top_p=0.9,          # Controls diversity of responses
            )

            # Extract the agent's response
            agent_response = response.choices[0].message.content.strip()

            # Add the agent's response to memory
            self.conversation_history.append({
                "role": "assistant",
                "conent": agent_response
            })

            return agent_response
        
        except Exception as e:
            error_message = f"I encountered an error while thinking: {str(e)}"
            self.console.print(f"[red]Error: {error_message}[/red]")
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
                self.console.print("[dim]🤔 Thinking...[/dim]")
                response = self.think(processed_input)

                # Agent responds
                self.act(response)

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            self.console.print("\n[yellow]Chat interrupted. Goodbye! 👋[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]And unexpected error occurred: {str(e)}[/red]")

# Function to create and run the agent
def main():
    """
    Main function to create and start the agent.
    """
    console = Console()

    # Is API key set?
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Error: OPEN_API_KEY not found in environment variables.[/red]")
        console.print("Please add your API key to the .env file.")
        return
    
    # Create and start the agen
    agent = SimpleAgent()
    agent.chat()

if __name__ == "__main__":
    main()