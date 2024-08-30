# standard imports
import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, List

# third-party imports
from dotenv import load_dotenv, find_dotenv
from langchain.globals import set_debug
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_community.vectorstores.faiss import FAISS
import openai

# local imports
from empirius import EmpiriusBot
from empirerift_knowledge_runnable import EmpireRiftKnowledgeRunnable
from wiki_scraper import get_all_wiki_docs

# Constants
SUBCOMMAND_RUN = 'run'
SUBCOMMAND_DOCS = 'docs'
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MODEL_TEMPERATURE = 0.3
DEFAULT_MODEL_TOP_P = 0.9
DEFAULT_INSTRUCTION_PROMPT_FILE = "instruction_prompt.txt"
DEFAULT_CONTEXT_PROMPT_FILE = "context_prompt.txt"
DEFAULT_MEMORIES_DIR = "memories"
DEFAULT_RETREIVER_K = 7
DEFAULT_PER_USER_HISTORY = False

# Helper functions
def read_file(path: Path) -> str:
    try:
        with path.open('r') as file:
            return file.read()
    except IOError as e:
        raise ValueError(f"Error reading file {path}: {e}")

def output_documents(documents: List[Document]):
    for i, doc in enumerate(documents):
        if i > 0:
            print("\n\n")
        print(f"Document {i}")
        print("-------------------")
        print(f"Content:\n{doc.page_content}")
        print("-------------------")
        print(f"Metadata:\n{doc.metadata}")
        print("-------------------\n\n")

def env_or_default(env_var: str, default: str) -> str:
    return os.getenv(env_var, default)
    
def bool_env_or_default(env_var: str, default: bool) -> bool:
    return os.getenv(env_var, str(default)).lower() == "true"

def load_mandatory_env_var(env_var: str) -> str:
    value = os.getenv(env_var)
    if not value:
        raise ValueError(f"{env_var} environment variable is required.")
    return value

def handle_arguments(config_args: List[str]) -> Tuple[argparse.Namespace, dict]:
    OPENAI_API_KEY = load_mandatory_env_var("OPENAI_API_KEY")
    DISCORD_TOKEN = load_mandatory_env_var("DISCORD_TOKEN")
    
    parser = argparse.ArgumentParser(description='A LLM-driven discord bot for answering questions from a knowledge base')
    subparsers = parser.add_subparsers(dest='command', help='Subcommand to run')
    
    run_command = subparsers.add_parser(SUBCOMMAND_RUN, help='run help')
    run_command.add_argument('--inst-prompt-file', type=str, default=DEFAULT_INSTRUCTION_PROMPT_FILE, help='File containing the instruction prompt')
    run_command.add_argument('--context-prompt-file', type=str, default=DEFAULT_CONTEXT_PROMPT_FILE, help='File containing the context prompt')
    run_command.add_argument('--memories-dir', type=str, default=DEFAULT_MEMORIES_DIR, help='Directory to store memories')
    run_command.add_argument('--openai-model', type=str, default=DEFAULT_MODEL, help='OpenAI model to use')
    run_command.add_argument('--temperature', type=float, default=DEFAULT_MODEL_TEMPERATURE, help='Model temperature')
    run_command.add_argument('--top-p', type=float, default=DEFAULT_MODEL_TOP_P, help='Model top-p')
    run_command.add_argument('--k', type=int, default=DEFAULT_RETREIVER_K, help='Number of retriever results to return')
    run_command.add_argument('--debug', action='store_true', help='Enable debug mode')
    run_command.add_argument('--local', action='store_true', help='Run the bot locally without discord')
    
    subparsers.add_parser(SUBCOMMAND_DOCS, help='docs help')

    config_args = parser.parse_args(config_args)
    mandatory_args = {'OPENAI_API_KEY': OPENAI_API_KEY, 'DISCORD_TOKEN': DISCORD_TOKEN}

    return config_args, mandatory_args

def run_knowledge_runnable_loop(runnable: EmpireRiftKnowledgeRunnable):
    while True:
        user_query = input("Du: ")
        if user_query.lower() == "exit":
            break
        session_id = "test"
        result = runnable.invoke(user_query, "test", session_id)
        print("-----------------------------------")
        print("Empirius: \n", result["answer"])
        print("-----------------------------------")

def main():
    load_dotenv(find_dotenv())
    config_args, mandatory_args = handle_arguments(sys.argv[1:])
    openai.api_key = mandatory_args['OPENAI_API_KEY']

    match config_args.command:
        case "docs":
            documents = get_all_wiki_docs()
            output_documents(documents)
            sys.exit()

        case "run":
            set_debug(config_args.debug)

            documents = get_all_wiki_docs()
            embedding = OpenAIEmbeddings()
            llm = ChatOpenAI(model=config_args.openai_model, temperature=config_args.temperature, top_p=config_args.top_p)
            retriever = FAISS.from_documents(documents=documents, embedding=embedding).as_retriever(search_kwargs={'k': config_args.k})
            get_instruction_prompt = lambda: read_file(Path(config_args.inst_prompt_file))
            get_context_prompt = lambda: read_file(Path(config_args.context_prompt_file))
            get_history = lambda session_id: FileChatMessageHistory(Path(config_args.memories_dir) / session_id)
            knowledge_runner = EmpireRiftKnowledgeRunnable(
                llm=llm, 
                get_context_prompt=get_context_prompt, 
                get_instruction_prompt=get_instruction_prompt, 
                retriever=retriever, 
                get_history=get_history
            )
            
            if config_args.local:
                run_knowledge_runnable_loop(knowledge_runner)
                sys.exit()

            EmpiriusBot(per_user=DEFAULT_PER_USER_HISTORY, runnable=knowledge_runner).run(mandatory_args['DISCORD_TOKEN'])
            sys.exit()

        case _:
            print("No subcommand specified. Exiting...")
            sys.exit(1)

if __name__ == '__main__':
    main()
