# standard imports
import os
import openai


# third-party imports
from dotenv import load_dotenv, find_dotenv

from langchain import globals as langchain_globals
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import RetrieverLike
from langchain_openai import OpenAIEmbeddings

# local imports
from empirius import EmpiriusBot
from wiki_scraper import get_wiki_page_urls, get_parsed_wiki_sections


# Constants
DEFAULT_WIKI_URL                    = "https://wiki.empirerift.com/"
# DEFAULT_MODEL                       = "gpt-3.5-turbo-0125"
DEFAULT_MODEL                       = "gpt-4o-mini"
DEFAULT_MODEL_TEMPERATURE           = 0.0
DEFAULT_INSTRUCTION_PROMPT_FILE     = "instruction_prompt.txt"
DEFAULT_CONTEXT_PROMPT_FILE         = "context_prompt.txt"
DEFAULT_MEMORIES_DIR                = "memories"
DEFAULT_RETREIVER_K                 = 7


# Helper functions
def load_instruction_prompt(prompt_file: str) -> str:
    with open(prompt_file, 'r') as file:
        instruction_prompt = file.read()
    return instruction_prompt

def output_documents(docs: list[Document]):
    for doc in docs:
        print(f"Content:\n{doc.page_content[:300]}")
        print("-------------------")
        print(f"Metadata:\n{doc.metadata}")
        print("-------------------\n"*3)
        
def create_faiss_retriever(docs: list[Document], k: int) -> RetrieverLike:
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={'k': k})
    return retriever



###################################################################################################################
# Main function                                                                                                   |
#                                                                                                                 |
# This function is the entry point for the bot. It loads environment variables, initializes the bot, and starts   |
# the bot.                                                                                                        |
#                                                                                                                 |
###################################################################################################################
# ---> Load config environment variables                                                                          |
#                                                                                                                 |
# | Variable Name       | Description                                                                             |
# |---------------------|-----------------------------------------------------------------------------------------|
# | OPENAI_API_KEY      | The OpenAI API key to use for the LLM. Required.                                        |
# | DISCORD_TOKEN       | The Discord bot token to use for the bot. Required.                                     |
# | WIKI_URL            | The URL of the wiki to scrape for documents. Defaults to https://wiki.empirerift.com/   |
# | PROMPT_FILE         | The file containing the instruction prompt. Defaults to prompt.txt                      |
# | MEMORIES_DIR        | The directory to store chat history. Defaults to memories/                              |
# | OPENAI_MODEL        | The OpenAI model to use for the LLM. Defaults to gpt-3.5-turbo-0125                     |
# | MODEL_TEMPERATURE   | The temperature to use for the LLM. Defaults to 0.3                                     |
# | RETREIVER_K         | The number of documents to retrieve from the retriever. Defaults to 5                   |
# | DEBUG_ENABLED       | Enable debug mode. Defaults to false                                                    |
# | VERBOSE             | Enable verbose mode. Defaults to false                                                  |
# | ONLY_DOCS           | Only load documents and exit. Defaults to false                                         |
###################################################################################################################

if __name__ == '__main__':
    # Load environment variables
    print("Loading environment variables from .env file...")
    load_dotenv(find_dotenv())

    OPENAI_API_KEY          = os.environ["OPENAI_API_KEY"]
    DISCORD_TOKEN           = os.environ["DISCORD_TOKEN"]
    
    if OPENAI_API_KEY is None:
        raise ValueError("OPENAI_API_KEY environment variable is required.")
    
    if DISCORD_TOKEN is None:
        raise ValueError("DISCORD_TOKEN environment variable is required.")
    
    def env_or_default(env_var: str, default: str) -> str:
        return os.environ[env_var] if env_var in os.environ else default
    
    def bool_env_or_default(env_var: str, default: bool) -> bool:
        return os.environ.get(env_var, str(default)).lower() == "true"
    
    WIKI_URL                = env_or_default("WIKI_URL", DEFAULT_WIKI_URL)
    INSTRUCTION_PROMPT_FILE = env_or_default("PROMPT_FILE", DEFAULT_INSTRUCTION_PROMPT_FILE)
    CONTEXT_PROMPT_FILE     = env_or_default("CONTEXT_PROMPT_FILE", DEFAULT_CONTEXT_PROMPT_FILE)
    MEMORIES_DIR            = env_or_default("MEMORIES_DIR", DEFAULT_MEMORIES_DIR)
    OPENAI_MODEL            = env_or_default("OPENAI_MODEL", DEFAULT_MODEL)
    MODEL_TEMPERATURE       = float(env_or_default("MODEL_TEMPERATURE", str(DEFAULT_MODEL_TEMPERATURE)))
    RETREIVER_K             = int(env_or_default("RETREIVER_K", str(DEFAULT_RETREIVER_K)))
    DEBUG                   = bool_env_or_default("DEBUG", False)
    VERBOSE                 = bool_env_or_default("VERBOSE", False)
    DEBUG_DOCS              = bool_env_or_default("DEBUG_DOCS", False)

    openai.api_key = OPENAI_API_KEY

    print("DEBUG mode enabled" if DEBUG else "DEBUG mode disabled")
    langchain_globals.set_debug(DEBUG)

    print("VERBOSE mode enabled" if VERBOSE else "VERBOSE mode disabled")
    langchain_globals.set_verbose(VERBOSE)
    
    print("Fetching wiki pages...")
    website_urls = get_wiki_page_urls(WIKI_URL)
    
    print("Fetching website URLs and parsing HTML...")
    documents = get_parsed_wiki_sections(website_urls, DEBUG)
    
    if DEBUG_DOCS:
        output_documents(documents)
        print("Only loading documents. Exiting...")
        exit()
        
    print("Creating retriever...")
    faiss_retriever = create_faiss_retriever(documents, k=RETREIVER_K)        

    # Run the bot
    print("Starting bot...")
    EmpiriusBot(
        instruction_prompt_fetcher=lambda: load_instruction_prompt(INSTRUCTION_PROMPT_FILE),
        context_prompt_fetcher=lambda: load_instruction_prompt(CONTEXT_PROMPT_FILE),
        vector_database_retriever=faiss_retriever,
        openai_model=OPENAI_MODEL,
        temperature=MODEL_TEMPERATURE,
        memories_dir=MEMORIES_DIR,
        per_user_history=False
    ).run(DISCORD_TOKEN)