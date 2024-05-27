from settings import ChatHistorySettings
from cache.chat_history import ChatHistory
from chains import RagChain, SimpleChain
from llm import Llm
from utils import my_logger
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages.ai import AIMessageChunk

class ChatAssistant:
    def __init__(self) -> None:
        self._logger = my_logger.set_logger(__name__)

    def respond(self, selected_model: str, message: str, is_rag: bool):
        llm = Llm().getinstance(selected_model)
        if is_rag:
            chain = RagChain().build(llm)
        else:
            chain = SimpleChain().build(llm)

        chain_with_message_history = RunnableWithMessageHistory(
            chain,
            ChatHistory().get_message_history,
            input_messages_key="input",
            history_messages_key=ChatHistorySettings.label
        )
        
        response = chain_with_message_history.stream(
            {"input": message},
            config={"configurable": {"session_id": "test"}}
        ) #TODO: session_idの振り方

        for msg in response:
            if isinstance(msg, AIMessageChunk):
                yield msg.content
            elif answer_chunk := msg.get("answer"):      
                yield answer_chunk
            else:
                pass