import os
from datetime import timedelta
from langchain_community.chat_message_histories.momento import MomentoChatMessageHistory

class ChatHistory():
    def __init__(self):
        self.cache_name = os.environ["MOMENTO_CACHE"]

    def get_message_history(self, session_id: str):
        return MomentoChatMessageHistory.from_client_params(
            session_id,
            self.cache_name,
            timedelta(hours=int(os.environ["MOMENTO_TTL"]))
        )