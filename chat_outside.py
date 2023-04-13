from langchain.vectorstores import Pinecone
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.llm import LLMChain
from langchain.llms.openai import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI

import yaml, pinecone
import streamlit as st


def chatOutside(query, model_name):

    with open('./credentials.yml', 'r') as file:
        cre = yaml.safe_load(file)
        # pinecone API
        pinecone_api_key = cre['pinecone']['apikey']
        # OpenAI API
        OPENAI_API_KEY = cre['openai']['apikey']

    # ------OpenAI: LLM---------------
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

    # ------OpenAI: Embed model-------------
    model_name = 'text-embedding-ada-002'
    embed = OpenAIEmbeddings(
        document_model_name=model_name,
        query_model_name=model_name,
        openai_api_key=OPENAI_API_KEY
    )
    # save this embeddings to numpy file
    # --- Pinecone ------
    pinecone.init(api_key=pinecone_api_key, environment="northamerica-northeast1-gcp")
    index_name = "outside-chatgpt"
    index = pinecone.Index(index_name)
    text_field = "text"
    vectorstore = Pinecone(index, embed.embed_query, text_field)


    #  ======= Langchain ChatDBQA with source chain =======
    qa = VectorDBQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        vectorstore=vectorstore
    )

    response = qa(query)
    return response