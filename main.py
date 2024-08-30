# standard imports
import os
import openai
import argparse
import datetime

# third-party imports
from dotenv import load_dotenv, find_dotenv
from sys import argv

from langchain.globals import set_debug
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_community.vectorstores.faiss import FAISS

# local imports
from empirius import EmpiriusBot
from empirerift_knowledge_runnable import EmpireRiftKnowledgeRunnable
from wiki_scraper import get_empirerift_wiki_sections, get_realisticseasons_wiki_sections, get_medieval_factions_wiki_sections


# Constants
SUBCOMMAND_RUN                      = 'run'
SUBCOMMAND_DOCS                     = 'docs'
DEFAULT_MODEL                       = "gpt-4o-mini"
DEFAULT_MODEL_TEMPERATURE           = 0.3
DEFAULT_MODEL_TOP_P                 = 0.9
DEFAULT_INSTRUCTION_PROMPT_FILE     = "instruction_prompt.txt"
DEFAULT_CONTEXT_PROMPT_FILE         = "context_prompt.txt"
DEFAULT_MEMORIES_DIR                = "memories"
DEFAULT_RETREIVER_K                 = 7
DEFAULT_PER_USER_HISTORY            = False

# Helper functions
def load_instruction_prompt(prompt_file: str) -> str:
    with open(prompt_file, 'r') as file:
        instruction_prompt = file.read()
    return instruction_prompt

def output_documents(docs: list[Document]):
    for doc in docs:
        print(f"Content:\n{doc.page_content}")
        print("-------------------")
        print(f"Metadata:\n{doc.metadata}")
        print("-------------------\n\n")
        
def create_faiss_retriever(docs: list[Document], k: int) -> VectorStoreRetriever:
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={'k': k})
    return retriever

def env_or_default(env_var: str, default: str) -> str:
        return os.environ[env_var] if env_var in os.environ else default
    
def bool_env_or_default(env_var: str, default: bool) -> bool:
    return os.environ.get(env_var, str(default)).lower() == "true"

def load_madatory_env_var(env_var: str) -> str:
    if env_var not in os.environ:
        raise ValueError(f"{env_var} environment variable is required.")
    return os.environ[env_var]

def handle_arguments(config_args: list[str]):
    
    OPENAI_API_KEY          = load_madatory_env_var("OPENAI_API_KEY")
    DISCORD_TOKEN           = load_madatory_env_var("DISCORD_TOKEN")
    INSTRUCTION_PROMPT_FILE = env_or_default("PROMPT_FILE", DEFAULT_INSTRUCTION_PROMPT_FILE)
    CONTEXT_PROMPT_FILE     = env_or_default("CONTEXT_PROMPT_FILE", DEFAULT_CONTEXT_PROMPT_FILE)
    MEMORIES_DIR            = env_or_default("MEMORIES_DIR", DEFAULT_MEMORIES_DIR)
    OPENAI_MODEL            = env_or_default("OPENAI_MODEL", DEFAULT_MODEL)
    MODEL_TEMPERATURE       = float(env_or_default("MODEL_TEMPERATURE", str(DEFAULT_MODEL_TEMPERATURE)))
    MODEL_TOP_P             = float(env_or_default("MODEL_TOP_P", str(DEFAULT_MODEL_TOP_P)))
    RETREIVER_K             = int(env_or_default("RETREIVER_K", str(DEFAULT_RETREIVER_K)))
    
    # Create the main parser
    parser = argparse.ArgumentParser(description='A LLM-driven discord bot for answering questions from a knowledge base')
    
    # Add subparsers
    subparsers = parser.add_subparsers(dest='command', help='Subcommand to run')

    # Subcommand 'run'
    run_command = subparsers.add_parser(SUBCOMMAND_RUN, help='run help')
    run_command.add_argument('--inst-prompt-file', type=str, default=INSTRUCTION_PROMPT_FILE, help='File containing the instruction prompt')
    run_command.add_argument('--context-prompt-file', type=str, default=CONTEXT_PROMPT_FILE, help='File containing the context prompt')
    run_command.add_argument('--memories-dir', type=str, default=MEMORIES_DIR, help='Directory to store memories')
    run_command.add_argument('--openai-model', type=str, default=OPENAI_MODEL, help='OpenAI model to use')
    run_command.add_argument('--temperature', type=float, default=MODEL_TEMPERATURE, help='Model temperature')
    run_command.add_argument('--top-p', type=float, default=MODEL_TOP_P, help='Model top-p')
    run_command.add_argument('--k', type=int, default=RETREIVER_K, help='Number of retriever results to return')
    run_command.add_argument('--debug', action='store_true', help='Enable debug mode')
    # Subcommand 'docs'
    subparsers.add_parser(SUBCOMMAND_DOCS, help='docs help')

    mandatory_args = {'OPENAI_API_KEY': OPENAI_API_KEY, 'DISCORD_TOKEN': DISCORD_TOKEN}
    config_args = parser.parse_args()

    return config_args, mandatory_args

def get_documents():
    empire_rift_docs = get_empirerift_wiki_sections()
    realistic_seasons_docs = get_realisticseasons_wiki_sections()
    medieval_factions_docs = get_medieval_factions_wiki_sections()
    documents = realistic_seasons_docs + medieval_factions_docs + empire_rift_docs
    return documents

def main():
    config_args, mandatory_args = handle_arguments(argv[1:])
    OPENAI_API_KEY = mandatory_args['OPENAI_API_KEY']
    DISCORD_TOKEN = mandatory_args['DISCORD_TOKEN']
    
    openai.api_key = OPENAI_API_KEY

    if config_args.command == SUBCOMMAND_DOCS:
        print('Fetching sections from EmpireRift wiki...')#
        documents = get_documents()
        output_documents(documents)
        exit()

    if config_args.command == SUBCOMMAND_RUN:
        set_debug(config_args.debug)

        llm                         = ChatOpenAI(model=config_args.openai_model, temperature=config_args.temperature, top_p=config_args.top_p)
        instruction_prompt_fetcher  = lambda: load_instruction_prompt(config_args.inst_prompt_file)
        context_prompt_fetcher      = lambda: load_instruction_prompt(config_args.context_prompt_file)
        history_fetcher             = lambda session_id: FileChatMessageHistory(f"{config_args.memories_dir}/{session_id}")
        documents                   = get_documents()
        faiss_retriever             = create_faiss_retriever(documents, config_args.k)        

        empirerift_knowledge_runnable = EmpireRiftKnowledgeRunnable(
            llm=llm,
            context_prompt_fetcher=context_prompt_fetcher,
            instruction_prompt_fetcher=instruction_prompt_fetcher,
            retriever=faiss_retriever,
            history_fetcher=history_fetcher,
        )

        # while True:
        #     user_query = input("Du: ")
        #     print("-----------------------------------")
        #     if user_query == "exit":
        #         break
        #     session_id = "test"
        #     result = empirerift_knowledge_runnable.invoke(user_query, "test", session_id)
        #     print("Empirius: ", result["answer"])
        #     print("-----------------------------------")

        print("Running Empirius bot...")
        empirius = EmpiriusBot(
            per_user_history=DEFAULT_PER_USER_HISTORY, 
            empirerift_knowledge_runnable=empirerift_knowledge_runnable
        )
        empirius.run(DISCORD_TOKEN)
        exit()
    
    print("No subcommand specified. Exiting...")
    exit(1)

if __name__ == '__main__':
    # Load environment variables
    print("Loading environment variables from .env file...")
    load_dotenv(find_dotenv())
    main()