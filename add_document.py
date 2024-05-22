import os
import sys

from utils import my_logger
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
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

if __name__ == "__main__":
    file_path = sys.argv[1] # コマンドライン引数を取得
    loader = UnstructuredPDFLoader(file_path)
    raw_docs = loader.load()
    logger.info("Loaded %d documents", len(raw_docs))

    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs = text_splitter.split_documents(raw_docs)
    logger.info("Split %d documents", len(docs))

    # 分割したドキュメントをVectorDBに読み込む
    vectorstore = initialize_vectorstore()
    vectorstore.add_documents(docs)