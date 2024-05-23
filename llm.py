from langchain_openai.chat_models import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from genai.client import Client
from genai.credentials import Credentials
from genai.extensions.langchain import LangChainChatInterface
from genai.schema import DecodingMethod, TextGenerationParameters

import settings

class Llm():
    def getinstance(self, model) -> BaseChatModel:
        """画面で選択されたモデル名からどのモデルを利用するかを判別するフラグを検索"""
        selected_model_flg = None
        for d in settings.MODEL_OPTIONS:
            if d["model"] == model:
                selected_model_flg = d["flg"]
                break
            else:
                selected_model_flg = settings.FLG_IBM

        """LLMのフラグに応じてインスタンス化"""
        llm = None
        if selected_model_flg == settings.FLG_OPENAI_CHAT:
            llm = ChatOpenAI(
                model=model,
                streaming=True
            )
        else:
            llm = LangChainChatInterface(
                model_id=model, 
                client=Client(credentials=Credentials.from_env()), 
                parameters=TextGenerationParameters(
                    decoding_method=DecodingMethod.GREEDY,
                    temperature=os.environ["GENAI_API_TEMPERATURE"],
                ),
                streaming=True
            )
        return llm