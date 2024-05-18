import os
import settings

from utils import my_logger
from datetime import timedelta
from genai.client import Client
from genai.credentials import Credentials
from genai.extensions.langchain import LangChainChatInterface
from genai.schema import (
    DecodingMethod,
    HumanMessage,
    TextGenerationParameters
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.momento import MomentoChatMessageHistory
from langchain_openai.chat_models import ChatOpenAI

from add_document import initialize_vectorstore


class ChatAssistant:
    def __init__(self) -> None:
        self._logger = my_logger.set_logger(__name__)


    def __model_initializer(self, selected_model):
        # 画面で選択されたモデル名からどのモデルを利用するかを判別するフラグを検索
        selected_model_flg = None
        for d in settings.MODEL_OPTIONS:
            if d["model"] == selected_model:
                selected_model_flg = d["flg"]
                break
            else:
                selected_model_flg = settings.FLG_IBM

        # LLMのフラグに応じてインスタンス化
        llm = None
        if selected_model_flg == settings.FLG_OPENAI_CHAT:
            llm = ChatOpenAI(
                model=selected_model,
                streaming=True
            )
        else:
            llm = LangChainChatInterface(
                model_id=selected_model, 
                client=Client(credentials=Credentials.from_env()), 
                parameters=TextGenerationParameters(
                    decoding_method=DecodingMethod.GREEDY,
                    temperature=os.environ["GENAI_API_TEMPERATURE"],
                ),
                streaming=True
            )
        return llm

    def respond(self, selected_model, message):
        messages = [HumanMessagePromptTemplate.from_template(template="{message}")]
        prompt = ChatPromptTemplate.from_messages(messages)
        llm = self.__model_initializer(selected_model)
        chain = prompt | llm
        response = chain.stream({"message": message})
        for msg in response:
            yield msg.content

    def retrieval_qa(self, selected_model, message):
        # VectorDBの初期化
        vectorstore = initialize_vectorstore()

        """
        会話履歴を踏まえた質問回答を実現させる
        [変更]
        1. Retrieverは最新の入力だけでなく、質問履歴全体を入力とする
        2. LLMの生成部分では全履歴を考慮する
        """
        # チャット履歴
        momento_chat_history = MomentoChatMessageHistory.from_client_params(
            "test", # TODO: 
            os.environ["MOMENTO_CACHE"],
            timedelta(hours=int(os.environ["MOMENTO_TTL"]))
        )
        momento_chat_history.add_user_message(message)

        # llmはchain共通
        llm = self.__model_initializer(selected_model)

        """
        history_aware_retriever
        最新の入力をチャット履歴に格納し、クエリを生成する
        """
        retriever = vectorstore.as_retriever()
        query_messages = [
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(template="{input}"),
            HumanMessagePromptTemplate.from_template(template="Given the above conversation, generate a search query to look up to get information relevant to the conversation"),
        ]
        prompt = ChatPromptTemplate.from_messages(query_messages)
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt) # LCELのラッパー

        """
        documents_chain
        """
        retrieval_qa_messages = [
            SystemMessagePromptTemplate.from_template(template="Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(template="{input}")
        ]
        retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages(retrieval_qa_messages)
        document_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt) # LCELのラッパー
        
        """
        question_answer_chain
        """
        qa_chain = create_retrieval_chain(retriever_chain, document_chain) # LCELのラッパー

        response = qa_chain.stream({
            "chat_history": momento_chat_history.messages,
            "input": message,
        })
        answer = ""
        for msg in response:
            if answer_msg := msg.get("answer"):
                answer += answer_msg
                yield answer_msg

        momento_chat_history.add_ai_message(answer)        
        self._logger.debug(answer)
