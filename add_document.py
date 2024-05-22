import os

from git import Repo
from utils import my_logger
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
logger = my_logger.set_logger(__name__)


def embeddings_factory():
    return OpenAIEmbeddings()


def initialize_vectorstore():
    index = os.environ["PINECONE_INDEX_NAME"]
    embeddings = embeddings_factory()
    return PineconeVectorStore.from_existing_index(index, embeddings)

"""
TODO: loader, splitterを切り替え可能にする
"""
def add_documents(repo_url):
    repo_path = "/home/dmurata/workspace/test-dir" # TODO: change
    repo = Repo.clone_from(repo_url, to_path=repo_path)

    loader = GenericLoader.from_filesystem(
        repo_path +  "/libs/core/langchain_core", # FIXME: specific-directory
        glob="**/*",
        suffixes=[".py"],
        exclude=["**/non-utf8-encoding.py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
    )
    # loader = UnstructuredPDFLoader(path)
    documents = loader.load()
    logger.info("Loaded %d documents", len(documents))

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=2000,
        chunk_overlap=200
    )
    # text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    texts = python_splitter.split_documents(documents)
    logger.info("Split %d documents", len(texts))

    vectorstore = initialize_vectorstore()
    vectorstore.add_documents(texts)


def create_retriever() -> VectorStoreRetriever:
    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        k=8
    )
    return retriever
