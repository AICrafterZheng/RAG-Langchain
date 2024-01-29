from langchain.embeddings import  AzureOpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch

from config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_BASE, AZURE_VECTOR_STORE_ADDRESS, AZURE_VECTOR_STORE_PWD, AZURE_VECTOR_STORE_INDEX_NAME, AZURE_EMBEDDING_DEPLOYMENT, AZURE_OPENAI_API_VERSION

vector_store_address: str =  f"https://{AZURE_VECTOR_STORE_ADDRESS}.search.windows.net/"
vector_store_password: str = AZURE_VECTOR_STORE_PWD

embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(azure_endpoint = AZURE_OPENAI_API_BASE, openai_api_key = AZURE_OPENAI_API_KEY, openai_api_version = AZURE_OPENAI_API_VERSION, azure_deployment=AZURE_EMBEDDING_DEPLOYMENT, chunk_size=1)
index_name: str = AZURE_VECTOR_STORE_INDEX_NAME
VectorStore: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)
