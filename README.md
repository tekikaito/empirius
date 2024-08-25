# empirius

LLM-driven discord bot for Empire Rift. It retrieves documents from the wiki and uses them to generate responses to user queries.

## Features

- LLM-driven chatbot
- Document retrieval from wiki
- Chat history
- Debug mode
- Verbose mode

## Badges

[![CC0 License](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)


## Run Locally

Clone the project

```bash
git clone git@github.com:tekikaito/empirius.git
```

Go to the project directory

```bash
cd empirius
```

Create pipenv environment and install dependencies

```bash
pipenv shell && pipenv install
```

Create a `.env` file in the root directory and add the following:

```bash
OPENAI_API_KEY=your_openai_api_key
DISCORD_TOKEN=your_discord_bot_token
```

Start the bot

```bash
python main.py
```

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

| Variable Name         | Description                                                                             |
|-----------------------|-----------------------------------------------------------------------------------------|
| `OPENAI_API_KEY`      | The OpenAI API key to use for the LLM. **Required**.                                    |
| `DISCORD_TOKEN`       | The Discord bot token to use for the bot. **Required**.                                 |
| `WIKI_URL`            | The URL of the wiki to scrape for documents. Defaults to https://wiki.empirerift.com/   |
| `PROMPT_FILE`         | The file containing the instruction prompt. Defaults to prompt.txt                      |
| `MEMORIES_DIR`        | The directory to store chat history. Defaults to memories/                              |
| `OPENAI_MODEL`        | The OpenAI model to use for the LLM. Defaults to gpt-3.5-turbo-0125                     |
| `MODEL_TEMPERATURE`   | The temperature to use for the LLM. Defaults to 0.3                                     |
| `RETREIVER_K`         | The number of documents to retrieve from the retriever. Defaults to 5                   |
| `DEBUG_ENABLED`       | Enable debug mode. Defaults to false                                                    |
| `VERBOSE`             | Enable verbose mode. Defaults to false                                                  |
| `DEBUG_DOCS`          | Only load documents and exit. Defaults to false                                         |

## License

[CC0 License](https://creativecommons.org/publicdomain/zero/1.0)