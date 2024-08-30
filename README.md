# Empirius: A LLM-Driven Discord Bot

Empirius is a powerful Discord bot designed to answer questions from a knowledge base using large language models. This bot leverages the capabilities of OpenAI's models, integrated with a FAISS-based vector store for efficient retrieval of relevant documents.

## Badges

[![CC0 License](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

## Features

- **LLM Integration**: Uses OpenAI's GPT models for generating high-quality responses.
- **FAISS Vector Store**: Efficient document retrieval based on vector embeddings.
- **Customizable Prompts**: Supports custom instruction and context prompts.
- **Session-Based Memory**: Maintains conversation context across sessions.
- **Local and Discord Modes**: Can run locally for testing or as a full-fledged Discord bot.

This command will print the content and metadata of all documents retrieved from the knowledge base.

## Environment Variables

Ensure the following environment variables are set either in your `.env` file or your environment:

- `OPENAI_API_KEY`: Your OpenAI API key (required).
- `DISCORD_TOKEN`: Your Discord bot token (required for Discord integration).

## Usage

### Installation

Clone the project

```bash
git clone git@github.com:tekikaito/empirius.git
```

Go to the project directory

```bash
cd empirius
```

Create pipenv environment

```bash
pipenv shell
```

Install dependencies

```bash
pipenv install
```

### Run Mode

To start the bot in run mode, which interacts with users on Discord:

```bash
python main.py run
```

Options for the `run` subcommand:

- `--inst-prompt-file`: Path to the instruction prompt file (default: `instruction_prompt.txt`)
- `--context-prompt-file`: Path to the context prompt file (default: `context_prompt.txt`)
- `--memories-dir`: Directory to store session memories (default: `memories`)
- `--openai-model`: OpenAI model to use (default: `gpt-4o-mini`)
- `--temperature`: Model temperature (default: `0.3`)
- `--top-p`: Model top-p (default: `0.9`)
- `--k`: Number of retriever results to return (default: `7`)
- `--debug`: Enable debug mode
- `--local`: Run the bot locally without Discord integration

### Docs Mode

To output the documents stored in the knowledge base:

```bash
python main.py docs
```

## Acknowledgements

This project uses the following libraries and tools:

- [LangChain](https://github.com/hwchase17/langchain) for LLM support.
- [FAISS](https://github.com/facebookresearch/faiss) for vector search.
- [dotenv](https://github.com/theskumar/python-dotenv) for managing environment variables.

## License

This project is licensed under the CC0 License. See the [LICENSE](LICENSE) file for details.
