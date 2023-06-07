from langchain.text_splitter import RecursiveCharacterTextSplitter
from data_loader import splitting, tiktoken_len
# If using terminal
from tqdm.auto import tqdm
# If using in Jupyter notebook
from tqdm.autonotebook import tqdm
from uuid import uuid4


def upsert(data, embed, index, model):

    batch_limit = 25

    texts = []
    metadatas = []

    text_splitter = splitting(data, model)

    for i, record in enumerate(tqdm(data)):
        # 1. Get metadata fields for this record
        metadata = {
            'id': str(record['id']),
            'source': record['title'],
            'content': record['content']
        }

        # 2. Create chunks from the record text
        record_texts = text_splitter.split_text(record["content"])
        # print(record_texts)
        print("------------------------",tiktoken_len(record_texts[0]))
        
        # 3. Create individual metadata dicts for each chunk
        record_metadatas = [{
            "chunk": j, "text": text, **metadata
        } for j, text in enumerate(record_texts)]
        
        # 4. Append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)

        print(len(texts))
        
        # 5. Special case: if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.embed_documents(texts)
            # print("--- upserting for record: ", i, " ---")
            try:
                index.upsert(vectors=zip(ids, embeds, metadatas))
            except Exception as e:
                print("Error in upserting", e)

            texts = []
            metadatas = []
        
    info = index.describe_index_stats()

    return texts, metadatas, info
    



    
# ---------------------------------- Alternative upsert function ----------------------------------

def upsert2(data, embed, index, model):
    batch_limit = 100

    texts = []
    metadatas = []
    text_splitter = splitting(data, model)

    for i, record in enumerate(tqdm(data)):
        # first get metadata fields for this record
        metadata = {
            'id': str(record['id']),
            'original_id': record['title'],
            'content': record['content']
        }
        # now we create chunks from the record text
        record_texts = text_splitter.split_text(record['content'])
        # create individual metadata dicts for each chunk
        record_metadatas = [{
            "chunk": j, "text": text, **metadata
        } for j, text in enumerate(record_texts)]
        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        print(i)
        # if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas))
            texts = []
            metadatas = []