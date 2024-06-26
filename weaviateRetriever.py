import weaviate
from langchain_openai import OpenAIEmbeddings
from llama_index.core import SimpleDirectoryReader
from langchain_weaviate.vectorstores import WeaviateVectorStore
from dotenv import load_dotenv
from path import Path
import os
load_dotenv()

weaviate_url = os.getenv("WEAVIATE_URL")
client = weaviate.Client(
    url=weaviate_url,
    auth_client_secret=weaviate.auth.AuthClientPassword(
        username = os.getenv("WCS_USERNAME"), 
        password = os.getenv("WCS_PASSWORD")  
    ),
)
def create_vectordatabase():
    client.schema.delete_all()
    schemas = client.schema.get()
    print(schemas)
    # Define the Schema object to use `text-embedding-3-small` on `title` and `content`, but skip it for `url`
    moodlebot_schema = {
        "class": "MoodleBot",
        "description": "A collection of documents for MoodleBot",
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "text2vec-openai": {
            "model": "ada",
            "modelVersion": "002",
            "type": "text"
            }
        },
        "properties": [{
            "name": "doc_id",
            "description": "Document ID",
            "dataType": ["text"],
            "moduleConfig": {
                "text2vec-openai": {"skip": True, "vectorizePropertyName": False}
            }
        },
        {
            "name": "file",
            "description": "Filename",
            "dataType": ["text"],
            "moduleConfig": {
                "text2vec-openai": {"skip": True, "vectorizePropertyName": False}
            }
        },
        {
            "name": "page",
            "description": "Page number",
            "dataType": ["text"],
            "moduleConfig": {
                "text2vec-openai": {"skip": True, "vectorizePropertyName": False}
            }
        },
        {
            "name": "content",
            "description": "Content of the Slide",
            "dataType": ["text"],
            "moduleConfig": {
                "text2vec-openai": {"skip": False, "vectorizePropertyName": False}
            }
        }]
    }

    client.schema.create_class(moodlebot_schema)
    schemas = client.schema.get()
    print(schemas)
    datas = []
    for path in Path(os.getenv("SELECTED_FILES")).iterdir():
        if path.suffix != ".pdf":
            continue

        print(f"Processing {path.name}...")
        reader = SimpleDirectoryReader(
            input_files=[path]
        )
        docs = reader.load_data()
        for d in docs:
            data = dict()
            data["doc_id"] = d.doc_id
            data["page"] = d.metadata["page_label"]
            data["file"] = d.metadata["file_name"]
            data["text"] = d.text
            datas.append(data)
                
    print(len(datas))

    with client.batch as batch:
        for data in datas:
            batch.add_data_object(data, "MoodleBot")

    count = client.data_object.get(class_name="MoodleBot")['totalResults']
    print(count)
    return

weaviate_client = weaviate.connect_to_custom(
    http_host=os.getenv("WEAVIATE_POD"),
    http_port=8080,
    http_secure=False,
    grpc_host=os.getenv("WEAVIATE_GRPC"),
    grpc_port=50051,
    grpc_secure=False,
    auth_credentials=weaviate.auth.AuthClientPassword(
        username = os.getenv("WCS_USERNAME"), 
        password = os.getenv("WCS_PASSWORD")  
    ),
)

def get_docs(prompt):
    embedding = OpenAIEmbeddings()
    weav = WeaviateVectorStore(client=weaviate_client, index_name ="MoodleBot", text_key="text", embedding=embedding)
    
    retriever = weav.as_retriever()
    docs = retriever.invoke(prompt)
    
    return docs
