import datetime

# discord imports
import discord
from discord.ext import commands

from empirerift_knowledge_runnable import EmpireRiftKnowledgeRunnable


class EmpiriusBot(commands.Bot):
    """
    A Discord bot that uses the Empirius conversational RAG chain to answer questions.
    """
    def __init__(self, per_user_history: bool, empirerift_knowledge_runnable: EmpireRiftKnowledgeRunnable):
        """
        Initialize the EmpiriusBot.\n\n
        `empirerift_rag_chain: RunnableBindingBase`  - The EmpireRiftKnowledgeRAGChain to use for the bot.\n
        `per_user_history: bool`                     - Whether to store chat history per user or not.\n
        
        """
        intents = discord.Intents.default()
        intents.message_content = True
        
        self.per_user_history = per_user_history
        self.empirius_knowledge_runnable = empirerift_knowledge_runnable
        
        super().__init__(command_prefix="!", intents=intents)
        
    def _is_usermessage_to_instance(self, message: discord.Message) -> bool:
        if self.user is None:
            return False
        return self.user.mentioned_in(message) and not message.author.bot
    
    def _invoke_empirerift_rag_chain(self, user_query: str, user_name: str, session_id: str) -> str:
        result = self.empirius_knowledge_runnable.invoke(user_query, user_name, session_id)
        return result["answer"]
    
    def _strip_mention(self, message: discord.Message) -> str:
        if self.user is None:
            return message.content
        return message.content.replace(f'<@{self.user.id}>', '').strip()
    
    # Event handlers
    async def on_ready(self):
        print(f'Logged in as {self.user}')

    async def on_message(self, message):
        if not self._is_usermessage_to_instance(message):
            return
        
        user_query = self._strip_mention(message)
        
        if not user_query:
            return
        
        user = message.author
        session_id = str(user.id) if self.per_user_history else "default"
        answer = self._invoke_empirerift_rag_chain(user_query, user.name, session_id)
        
        await message.channel.send(answer)
