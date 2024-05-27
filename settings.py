ROLE_USER="user"
ROLE_ASSISTANT="assistant"
FLG_IBM="IBM"
FLG_OPENAI_CHAT="ChatOpenAI"
MODEL_OPTIONS=[
    {"model": "meta-llama/llama-3-70b-instruct", "flg": FLG_IBM},
    {"model": "gpt-3.5-turbo-0125", "flg": FLG_OPENAI_CHAT}
]
class ChatHistorySettings():
    label: str = "chat_history"