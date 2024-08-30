from typing import Callable
# from pydantic import BaseModel 
import datetime

# langchain & openai imports
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.base import  Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI


class EmpireRiftKnowledgeRunnable():
    llm: ChatOpenAI
    instruction_prompt: str
    context_prompt: str
    retriever: BaseRetriever
    history_fetcher: Callable[[str], BaseChatMessageHistory]
    runnable: RunnableWithMessageHistory
    
    """    
    A RAG chain that can be used to answer questions about the Empire Rift wiki.
    """
    def __init__(
        self,
        llm: ChatOpenAI,
        instruction_prompt_fetcher: Callable[[], str],
        context_prompt_fetcher: Callable[[], str],
        retriever: BaseRetriever,
        history_fetcher: Callable[[str], BaseChatMessageHistory],
    ):
        """
        Initialize the EmpireRiftKnowledgeRAGChain.\n\n
        `instruction_prompt_fetcher`            - A function that returns the instruction prompt.\n
        `context_prompt_fetcher`                - A function that returns the context prompt.\n
        `retriever: RetrieverLike`              - The retriever to use for the bot.\n
        `llm: ChatOpenAI`                       - The OpenAI language model to use for the bot.\n
        `per_user_history: bool`                - Whether to store chat history per user or not.\n
        """
        self.instruction_prompt = instruction_prompt_fetcher()
        self.context_prompt = context_prompt_fetcher()
        self.retriever = retriever
        self.llm = llm

        rag_chain = create_retrieval_chain(
            combine_docs_chain=self._create_docs_qa_chain(),
            retriever=self._create_history_aware_retriever(),
        )

        self.runnable = RunnableWithMessageHistory(
            rag_chain,
            get_session_history=history_fetcher,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    
    def _create_history_aware_retriever(self) -> Runnable:
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"\n{self.instruction_prompt}"),
                MessagesPlaceholder("chat_history"),
                ("human", "\n{input}"),
            ]
        )
        
        return create_history_aware_retriever(   
            self.llm,
            self.retriever,
            contextualize_q_prompt
        )
        
    def _create_docs_qa_chain(self) -> Runnable:
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"\n{self.instruction_prompt}"),
                MessagesPlaceholder("chat_history"),
                ("system", f"\n{self.context_prompt}"),
                ("human", "\n{input}"),
            ]
        )
        return create_stuff_documents_chain(self.llm, qa_prompt)
    
    def invoke(self, user_query: str, user_name: str, session_id: str) -> str:
        current_iso_time = datetime.datetime.now().isoformat()
        return self.runnable.invoke(
            {"input": user_query, "time": current_iso_time, "user": user_name},
            config={"configurable": {"session_id": session_id}},
        )