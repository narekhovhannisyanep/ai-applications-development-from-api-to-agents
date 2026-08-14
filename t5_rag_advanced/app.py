from commons.constants import (
    GPT_5_4_NANO,
    OPENAI_API_KEY,
    OPENAI_CHAT_COMPLETIONS_ENDPOINT,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_EMBEDDINGS_ENDPOINT,
)
from commons.models.conversation import Conversation
from commons.models.message import Message
from commons.models.role import Role
from t5_rag_advanced.chat.chat_completion_client import ChatCompletionClient
from t5_rag_advanced.embeddings.embeddings_client import EmbeddingsClient
from t5_rag_advanced.embeddings.text_processor import SearchMode, TextProcessor

SYSTEM_PROMPT = """
You are a RAG-powered Microwave Manual Assistant.

Structure:
<RAG Context>
Retrieved RAG context

<User Message>
The actual user's message.

Instructions:
- Use <RAG Context> when answering to the <User Message>.
- Answer only based on the <RAG Context> or conversation history.
- If no relevant information exists in the <RAG Context> or conversation history, state that you can not answer the question.
"""

USER_PROMPT = """
<RAG Context>
{}

<User Message>
{}
"""

embeddings_client = EmbeddingsClient(
    OPENAI_EMBEDDINGS_ENDPOINT, OPENAI_EMBEDDING_MODEL, OPENAI_API_KEY
)
chat_completions_client = ChatCompletionClient(
    OPENAI_CHAT_COMPLETIONS_ENDPOINT, GPT_5_4_NANO, OPENAI_API_KEY
)
db_config = {
    "host": "localhost",
    "port": "5433",
    "database": "vectordb",
    "user": "postgres",
    "password": "postgres",
}
text_processor = TextProcessor(embeddings_client, db_config)
conversation = Conversation()
conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))

while True:
    user_input = input("\n🗿: ")
    if user_input.lower().strip() == "exit":
        break
    raw_context = text_processor.search(
        SearchMode.EUCLIDIAN_DISTANCE, user_input, 10, 1.5, 384
    )
    context = "\n\n".join(c.get("text") for c in raw_context)
    formatted_user_input = USER_PROMPT.format(context, user_input)
    conversation.add_message(Message(role=Role.USER, content=formatted_user_input))
    llm_message = chat_completions_client.get_completion(conversation.get_messages())
    print("\n🤖: ", llm_message.content)
    conversation.add_message(Message(Role.ASSISTANT, llm_message.content))
