import json
from typing import List, TypedDict
from langchain.text_splitter import TextSplitter
from langchain_ollama import OllamaEmbeddings


class Chunk(TypedDict):
    text: str
    f10kItem: str
    chunkSeqId: str
    formId: str
    chunkId: str
    names: List[str]
    cik: str
    cusip6: str
    source: str


class Embedder:
    embedder = OllamaEmbeddings(
        model="llama3",
    )

    @classmethod
    def embed_text(self, text: str) -> List[float]:
        return self.embedder.embed_query(text)


def split_form10k_data_from_file(
    file: str, section_list: List[str], text_splitter: TextSplitter
) -> List[Chunk]:
    chunks_with_metadata = []  # use this to accumlate chunk records
    file_as_object = json.load(open(file))  # open the json file

    for item in section_list:  # pull these keys from the json
        item_text = file_as_object[item]  # grab the text of the item
        item_text_chunks = text_splitter.split_text(
            item_text
        )  # split the text into chunks

        for index, chunk in enumerate(
            item_text_chunks[:20]
        ):  # only take the first 20 chunks
            form_id = file[
                file.rindex("/") + 1 : file.rindex(".")
            ]  # extract form id from file name
            # finally, construct a record with metadata and the chunk text

            chunks_with_metadata.append(
                {
                    "text": chunk,
                    # metadata from looping...
                    "f10kItem": item,
                    "chunkSeqId": index,
                    # constructed metadata...
                    "formId": f"{form_id}",  # pulled from the filename
                    "chunkId": f"{form_id}-{item}-chunk{index:04d}",
                    # metadata from file...
                    "names": file_as_object["names"],
                    "cik": file_as_object["cik"],
                    "cusip6": file_as_object["cusip6"],
                    "source": file_as_object["source"],
                }
            )
    return chunks_with_metadata
