import json
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

tokenizer = tiktoken.get_encoding('cl100k_base')

# ------------ importing the data --------------------

def load_data(file_path):

    # Opening JSON file
    with open(file_path) as json_file:
        data = json.load(json_file)

    return data


# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def splitting(data, model):

    # ------------ Splitting content into smaller chunks --------------------
    tiktoken.encoding_for_model(model)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )

    return text_splitter