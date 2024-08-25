import datetime
from typing import Callable

# discord imports
import discord
from discord.ext import commands

# langchain & openai imports
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.base import RunnableBindingBase
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import RetrieverLike
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI

DEFAULT_GLOBAL_SESSION_ID = "session"

class EmpiriusBot(commands.Bot):
    """
    A Discord bot that uses the Empirius conversational RAG chain to answer questions.
    """
    def __init__(
        self,
        context_prompt_fetcher: Callable[[], str],
        instruction_prompt_fetcher: Callable[[], str],
        vector_database_retriever: RetrieverLike,
        openai_model: str,
        temperature: float,
        memories_dir: str,
        per_user_history: bool = True,
    ):
        """
        Initialize the EmpiriusBot.\n\n
        `context_prompt_fetcher`            - A function that returns the context prompt.\n
        `instruction_prompt_fetcher`        - A function that returns the instruction prompt.\n
        `faiss_retriever: RetrieverLike`    - The retriever to use for the bot.\n
        `model: str`                        - The OpenAI model to use for the bot.\n
        `temperature: float`                - The temperature to use for the bot.\n
        `memories_dir: str`                 - The directory to store chat history.\n
        `per_user_history: bool`            - Whether to store chat history per user or not.\n
        """
        intents = discord.Intents.default()
        intents.message_content = True  # Enable the intents you need

        super().__init__(command_prefix="!", intents=intents)
        
        self.openai_model = openai_model
        self.temperature = temperature
        self.per_user_history = per_user_history
        self.memories_dir = memories_dir
        self.instructions_prompt_fetcher = instruction_prompt_fetcher
        self.context_prompt_fetcher = context_prompt_fetcher
        self.vector_database_retriever = vector_database_retriever
        self.conversational_rag_chain = self._initialize_rag_chain()
    
    def _create_openai_llm(self) -> ChatOpenAI:
        return ChatOpenAI(model=self.openai_model, temperature=self.temperature)
        
    def _create_get_session_history_fn(self, directory: str) -> BaseChatMessageHistory:
        return lambda session_id: FileChatMessageHistory(f"{directory}/{session_id}")
    
    def _create_history_aware_retriever(self, llm: ChatOpenAI, retriever: RetrieverLike, instruction_prompt: str) -> RetrieverLike:
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"\n{instruction_prompt}"),
                MessagesPlaceholder("chat_history"),
                ("human", "\n{input}"),
            ]
        )
        return create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    def _create_qa_chain(self, llm: ChatOpenAI, instruction_prompt: str, context_prompt: str) -> RunnableBindingBase:
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"\n{instruction_prompt}"),
                MessagesPlaceholder("chat_history"),
                ("system", f"\n{context_prompt}"),
                ("human", "\n{input}"),
            ]
        )
        return create_stuff_documents_chain(llm, qa_prompt)
    
    def _initialize_rag_chain(self) -> RunnableBindingBase:
        """
        Initialize the RAG chain for the bot. This chain will be used to answer questions.
        """
        # Initialize LLM and vectorstore
        llm = self._create_openai_llm()
        
        # Create chat prompts
        instruction_prompt = self.instructions_prompt_fetcher()
        context_prompt = self.context_prompt_fetcher()
        
        # Create RAG chain
        rag_chain = create_retrieval_chain(
            combine_docs_chain=self._create_qa_chain(llm, instruction_prompt, context_prompt),
            retriever=self._create_history_aware_retriever(llm, self.vector_database_retriever, instruction_prompt),
        )
        
        return RunnableWithMessageHistory(
            rag_chain,
            get_session_history=self._create_get_session_history_fn(self.memories_dir),
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
    def _is_usermessage_to_instance(self, message: discord.Message) -> bool:
        return self.user.mentioned_in(message) and not message.author.bot
    
    def _invoke_conversational_rag_chain(self, user_query: str, user: discord.User, session_id: str) -> str:
        current_iso_time = datetime.datetime.now().isoformat()
        result = self.conversational_rag_chain.invoke(
            {"input": user_query, "time": current_iso_time, "user": user.name},
            config={"configurable": {"session_id": session_id}},
        )
        return result["answer"]
    
    def _strip_mention(self, message: discord.Message) -> str:
        return message.content.replace(f'<@{self.user.id}>', '').strip()
    
    
    ########################################################
    # Event handlers                                       #
    ########################################################
    
    async def on_ready(self):
        print(f'Logged in as {self.user}')

    async def on_message(self, message):
        if not self._is_usermessage_to_instance(message):
            return
        
        user = message.author
        user_query = self._strip_mention(message)
        session_id = str(user.id) if self.per_user_history else DEFAULT_GLOBAL_SESSION_ID
        
        if not user_query:
            return
        
        answer = self._invoke_conversational_rag_chain(user_query, user, session_id)
        await message.channel.send(answer)
