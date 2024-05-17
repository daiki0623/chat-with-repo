ROLE_USER="user"
ROLE_ASSISTANT="assistant"
FLG_IBM="IBM"
FLG_OPENAI_CHAT="ChatOpenAI"
MODEL_OPTIONS=[ # アプリ用モデル一覧
    {"model": "meta-llama/llama-3-70b-instruct", "flg": "IBM"},
    {"model": "gpt-3.5-turbo-16k-0613", "flg": "ChatOpenAI"}
]
MODEL_OPTIONS_UI=tuple(d["model"] for d in MODEL_OPTIONS) # UI用にモデル一覧