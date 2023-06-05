from data_loader import load_data
from vector_database import embed_text 
from upsert import upsert, upsert2
from chat_outside import chatOutside

def main():

    # data_loader.py
    data = load_data('2022_page_summaries2.json')

    # vector_database.py
    pinecone_index, embed = embed_text(index_name = 'outside-chatgpt', model_name = 'text-embedding-ada-002', environment="northamerica-northeast1-gcp")

    print(embed)

    # # upsert.py
    texts, metadatas, info = upsert(data, embed, pinecone_index, model = 'gpt-3.5-turbo')
    print(info)

    # # # chat_outside.py
    # query = "How did silicon valley bank collapse?"
    # qa = chatOutside(query, model_name = 'text-embedding-ada-002')
    # print(qa)

    

if __name__ == '__main__':
    main()
