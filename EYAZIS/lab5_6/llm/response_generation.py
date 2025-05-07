from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResponseGenerator:
    def __init__(self):
        """
        Initialize the response generator with Ollama LLM.
        Uses the gemma3:1b model by default (small, efficient model good for chat)
        """
        try:
            # Initialize the LLM with temperature for some creativity
            self.llm = Ollama(
                model="gemma3:1b",  # You can change to "llama3" or other models
                temperature=0.7,
                top_p=0.9,
                num_ctx=2048  # Context window size
            )

            # Set up the prompt template
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are a knowledgeable Music Assistant.
                 Provide focused, helpful answers about music, genres, history and music theory.

                Key requirements:
                1. Keep responses concise (2-4 sentences)
                2. Match the user's sentiment in your tone
                3. Strictly follow the provided context
                """),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ])

            # Initialize conversation memory
            self.memory = ConversationBufferMemory(
                memory_key="history",
                return_messages=True,
                input_key="input"
            )

            # Create conversation chain
            self.conversation = ConversationChain(
                llm=self.llm,
                memory=self.memory,
                prompt=self.prompt,
                input_key="input",
                verbose=False
            )

            logger.info("Ollama response generator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {str(e)}")
            raise

    def generate_response(self, user_input: str, context: Dict) -> Optional[str]:
        """
        Generate a response using Ollama LLM with conversation context.

        Args:
            user_input: The user's message
            context: Dictionary containing analysis results (sentiment, keywords, etc.)

        Returns:
            Generated response string or None if failed
        """
        try:
            # Enhance the user input with context
            enhanced_input = (
                f"User message (sentiment: {context.get('sentiment', 'neutral')}): {context.get('weighted_query', user_input)}\n"
                f"Key topics: {', '.join(context.get('keywords', []))}\n"
                f"Context from database: {context.get('context', 'No specific context')}"
            )

            logger.info(f"Generating response for: {enhanced_input}")

            # Invoke the conversation chain
            response = self.conversation.invoke({"input": enhanced_input})

            # Get the generated response
            bot_response = response['response'].strip()

            logger.info(f"Generated response: {bot_response[:100]}...")
            return bot_response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'm really into music! What are you listening to these days?"

    def __call__(self, user_input: str, context: Dict) -> Optional[str]:
        """Convenience method to call generate_response directly."""
        return self.generate_response(user_input, context)


# Example usage
if __name__ == "__main__":
    # Initialize the generator
    generator = ResponseGenerator()

    # Example context from text analysis
    example_context = {
        'sentiment': 'positive',
        'keywords': ['rock', 'concert', 'band'],
        'context': 'The user enjoys rock music and live performances.'
    }

    # Example conversation
    user_messages = [
        "I love going to rock concerts!",
        "What are some good rock bands to see live?",
        "Tell me about Metallica"
    ]

    print("Music Chatbot Demo (using Ollama)")
    print("Type 'quit' to exit\n")

    for message in user_messages:
        print(f"You: {message}")
        response = generator(message, example_context)
        print(f"Bot: {response}\n")