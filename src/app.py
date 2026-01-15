import os
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from paths import DATA_DIR, APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR
#from utils import load_yaml_config
from prompt_template import template
#from langchain_core.runnables import RunnablePassthrough
import logging

logger = logging.getLogger("Project1")

def setup_logging():

    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(os.path.join(OUTPUTS_DIR, "rag_assistant.log"))
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #file_handler.setFormatter(formatter)
    #console_handler.setFormatter(formatter)

# Load environment variables
load_dotenv()


def load_documents() -> List[str]:
    """
    Load documents for demonstration.

    Returns:
        List of sample documents
    """
    results = []
    # TODO: Implement document loading
    # HINT: Read the documents from the data directory
    # HINT: Return a list of documents
    # HINT: Your implementation depends on the type of documents you are using (.txt, .pdf, etc.)

    # Your implementation here

    documents_path = DATA_DIR
    documents = []

    logger.info(f"documents_path: {documents_path}")

    for file in os.listdir(documents_path):
        if file.endswith(".txt"):
            file_path = os.path.join(documents_path, file)
            try:
                loader = TextLoader(file_path, autodetect_encoding=True)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                logger.info(f"Successfully loaded: {file}")
            except Exception as e:
                logger.warning(f"Error loading {file}: {str(e)}")
        elif file.endswith(".pdf"):
            file_path = os.path.join(documents_path, file)
            try:
                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                logger.info(f"Successfully loaded: {file}")
            except Exception as e:
                logger.warning(f"Error loading {file}: {str(e)}")
    
    logger.info(f"\nTotal documents loaded: {len(documents)}")

    for doc in documents:
        results.append(doc.page_content)
    
    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()
        self.vector_db.logger = logger

        # Create RAG prompt template
        # TODO: Implement your RAG prompt template
        # HINT: Use ChatPromptTemplate.from_template() with a template string
        # HINT: Your template should include placeholders for {context} and {question}
        # HINT: Design your prompt to effectively use retrieved context to answer questions
        #self.prompt_template = None 
        # Your implementation here
        
        template_string = template
        logger.info(f"prompt template: {template_string}")

        self.prompt_template = ChatPromptTemplate.from_template(template_string) 
        #self.prompt_template = ChatPromptTemplate.from_template("Relevant documents:\n\n{context}\n\nUser's question:\n\n{question}") 
        
        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        #self.chain = ({"context": RunnablePassthrough(), "question": RunnablePassthrough()} | self.llm | StrOutputParser()) 
        #self.chain.invoke()

        logger.info("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            logger.info(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            logger.info(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            logger.info(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)

    def invoke(self, input: str, n_results: int = 5) -> str:
        """
        Query the RAG assistant.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            Dictionary containing the answer and retrieved context
        """
        llm_answer = ""
        # TODO: Implement the RAG query pipeline
        # HINT: Use self.vector_db.search() to retrieve relevant context chunks
        # HINT: Combine the retrieved document chunks into a single context string
        # HINT: Use self.chain.invoke() with context and question to generate the response
        # HINT: Return a string answer from the LLM

        logger.debug("ready to call db...")
        results = self.vector_db.search(input, n_results)
        logger.debug(f"results: {len(results)}")

        prompt_value = self.prompt_template.format_messages(context = results["documents"], question = input)
        logger.debug(f"prompt_value = {prompt_value}")
        llm_answer = self.llm.invoke(prompt_value).content

        #llm_answer = self.llm.invoke(input_data).content
        #llm_answer = self.llm.invoke(results["documents"], input).content

        # Your implementation here
        return llm_answer

    def query(self, query: str) -> str:
        logger.debug(f"query = {query}")
        return self.invoke(query, 10)

def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        setup_logging()


        # Initialize the RAG assistant
        logger.info("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        logger.info("\nLoading documents...")
        sample_docs = load_documents()
        logger.info(f"Loaded {len(sample_docs)} sample documents")

        assistant.add_documents(sample_docs)

        done = False

        while not done:
            question = input("Enter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            elif question == "":
                done = False
            else:
                result = assistant.query(question)
                logger.info(result)

    except Exception as e:
        logger.warning(f"Error running RAG assistant: {e}")
        logger.warning("Make sure you have set up your .env file with at least one API key:")
        logger.warning("- OPENAI_API_KEY (OpenAI GPT models)")
        logger.warning("- GROQ_API_KEY (Groq Llama models)")
        logger.warning("- GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    main()
