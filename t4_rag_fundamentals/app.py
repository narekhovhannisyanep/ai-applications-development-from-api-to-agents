import os
import pathlib

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

from commons.constants import GPT_5_4_NANO, OPENAI_API_KEY

_SYSTEM_PROMPT = """
You are a RAG-powered assistant that assists users with their questions about microwave usage.

## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer only based on the RAG context or conversation history.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you can not answer the question.
"""

_USER_PROMPT = """
##RAG CONTEXT:
{context}

##USER QUESTION:
{query}"""


class MicrowaveRAG:
    def __init__(self, embeddings: OpenAIEmbeddings, llm_client: ChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """
        Load existing FAISS index from disk or create a new one.
        Returns:
              VectorStore: Initialized FAISS vectorstore.
        """
        print("🔄 Initializing Microwave Manual RAG System...")

        if os.path.exists("micorwave_faiss_index"):
            vectorstore = FAISS.load_local(
                folder_path="microwave_faiss_index",
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print("✅ Loaded existing FAISS index.")
        else:
            vectorstore = self._create_new_index()
            print("✅ RAG system initialized successfully!")

        return vectorstore

    def _create_new_index(self) -> VectorStore:
        """
        Load the manual, split into chunks, embed, and save a new FAISS index.
        Returns:
              VectorStore: Newly created and saved FAISS vectorstore.
        """
        current_dir = pathlib.Path(__file__).resolve().parent
        microwave_manual_path = current_dir / "microwave_manual.txt"
        microwave_faiss_index_path = current_dir / "microwave_faiss_index"

        print("📖 Loading text document...")
        loader = TextLoader(file_path=microwave_manual_path, encoding="utf-8")
        documents = loader.load()

        print("✂️ Splitting the document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", "."]
        )
        chunks = text_splitter.split_documents(documents=documents)
        print(f"✅ Created {len(chunks)} chunks.")

        print("🔍 Creating embeddings and FAISS index...")
        vectorstore = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
        vectorstore.save_local(str(microwave_faiss_index_path))
        print("💾 Index saved for future use.")

        return vectorstore

    def retrieve_context(self, query: str, k: int = 4, score=0.3):
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        print("=" * 100)
        print("🔍 STEP 1: RETRIEVAL")
        print(f"{'-' * 100}")
        print(f"Query: '{query}'")
        print(
            f"Searching for top {k} most relevant chunks with similarity score {score}:"
        )

        document_page_contents = []

        similarity_search_result = (
            self.vectorstore.similarity_search_with_relevance_scores(
                query=query, k=k, score_threshold=score
            )
        )

        for document, relevance_score in similarity_search_result:
            document_page_contents.append(document.page_content)
            print(f"\n--- (Relevance Score: {relevance_score:.3f}) ---")
            print(f"Content: {document.page_content}")

        print("=" * 100)

        return "\n\n".join(document_page_contents)

    def augment_prompt(self, query: str, context: str):
        """
        Inject retrieved context and user query into the prompt template.
        Args:
              query (str): The user's question.
              context (str): Retrieved context from the vectorstore.
        Returns:
              str: Formatted prompt ready for the LLM.
        """
        print(f"\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")
        augmented_prompt = _USER_PROMPT.format(context=context, query=query)
        print(f"{augmented_prompt}\n{'=' * 100}")
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str):
        """
        Send the augmented prompt to the LLM and return its response.
        Args:
              augmented_prompt (str): The prompt with injected context and query.
        Returns:
              str: The LLM-generated answer.
        """
        input_messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt),
        ]

        ai_message = self.llm_client.invoke(input=input_messages)
        print(f"\n🤖 STEP 3: GENERATION\n{'-' * 100}")
        print(f"{ai_message.content}\n{'=' * 100}")
        return ai_message.content


def main(rag: MicrowaveRAG):
    print("🎯 Microwave RAG Assistant")

    while True:
        user_input = input("\n>").strip()
        if user_input.lower() == "exit":
            break

        context = rag.retrieve_context(user_input)
        augmented_prompt = rag.augment_prompt(query=user_input, context=context)
        ai_message = rag.generate_answer(augmented_prompt=augmented_prompt)
        print(ai_message)


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", api_key=SecretStr(OPENAI_API_KEY)
)
openai_client = ChatOpenAI(
    model=GPT_5_4_NANO, api_key=SecretStr(OPENAI_API_KEY), temperature=0.0
)
main(MicrowaveRAG(embeddings=embeddings, llm_client=openai_client))
