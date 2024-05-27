from abc import ABC, abstractmethod
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from add_document import create_retriever
from prompt import QUERY_SYS_PROMPT, RETRIEVAL_SYS_PROMPT
from settings import ChatHistorySettings

class Chain(ABC):
    @abstractmethod
    def build(self, llm: BaseChatModel):
        pass

class SimpleChain(Chain):
    def build(self, llm: BaseChatModel):
        messages = [
            MessagesPlaceholder(variable_name=ChatHistorySettings.label),
            HumanMessagePromptTemplate.from_template(template="{input}")
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm
        return chain

class RagChain(Chain):
    def build(self, llm: BaseChatModel):
        """history_aware_retriever:最新の入力をチャット履歴に格納し、クエリを生成する"""
        # retriever = create_pdf_retriever()
        retriever = create_retriever()
        query_messages = [
            MessagesPlaceholder(variable_name=ChatHistorySettings.label),
            HumanMessagePromptTemplate.from_template(template="{input}"),
            HumanMessagePromptTemplate.from_template(template=QUERY_SYS_PROMPT),
        ]
        prompt = ChatPromptTemplate.from_messages(query_messages)
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt) # LCELのラッパー

        """documents_chain"""
        retrieval_qa_messages = [
            SystemMessagePromptTemplate.from_template(template=RETRIEVAL_SYS_PROMPT),
            MessagesPlaceholder(variable_name=ChatHistorySettings.label),
            HumanMessagePromptTemplate.from_template(template="{input}")
        ]
        retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages(retrieval_qa_messages)
        document_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt) # LCELのラッパー
        
        """question_answer_chain"""
        qa_chain = create_retrieval_chain(retriever_chain, document_chain) # LCELのラッパー
        return qa_chain