from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader, DirectoryLoader, TextLoader
from constants import fileName
from acs import VectorStore



print("Loading data...")
text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader("./data/ICMs", glob="**/*.txt", show_progress=True, loader_kwargs=text_loader_kwargs, loader_cls=TextLoader)
#loader = UnstructuredFileLoader(f"./data/{fileName}.txt")

raw_documents = loader.load()
print(f"Loaded {len(raw_documents)} documents")
print(raw_documents[0])

print("Splitting text...")
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=600,
    chunk_overlap=100,
    length_function=len,
)
docs = text_splitter.split_documents(raw_documents)

print("Creating vectorstore...")

VectorStore.add_documents(docs)