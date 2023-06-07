
# ------------ Embedding the data --------------------
import yaml
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

def embed_text(index_name, model_name, environment):
    # 0. Initialize Pinecone Client
    with open('./credentials.yml', 'r') as file:
        cre = yaml.safe_load(file)
        # pinecone API
        pinecone_api_key = cre['pinecone']['apikey']
        # OpenAI API
        OPENAI_API_KEY = cre['openai']['apikey']

    pinecone.init(api_key=pinecone_api_key, environment=environment)

    # 1. Create a new index
    index_name = index_name

    # 2. Use OpenAI's ada-002 as embedding model
    model_name = model_name
    embed = OpenAIEmbeddings(
        document_model_name=model_name,
        query_model_name=model_name,
        openai_api_key=OPENAI_API_KEY
    )
    embed_dimension = 1536

    # 3. check if index already exists (it shouldn't if this is first time)
    if index_name not in pinecone.list_indexes():
        # if does not exist, create index
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=embed_dimension
        )

    # 3. Connect to index
    index = pinecone.Index(index_name)

    return index, embed

