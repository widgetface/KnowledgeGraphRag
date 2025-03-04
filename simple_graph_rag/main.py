from utils.splitter_utils import split_form10k_data_from_file
from utils.cyphers import (
    delete_all_cypher,
    unique_constraint_cypher,
    merge_chunk_node_cypher,
    count_nodes_cypher,
    create_vector_index,
    show_vector_indices,
)
from utils.pipeline import ReviewDatasetPipeline
from utils.prompts import template, short_template
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import neo4j

from time import sleep

DATA_FILE_PATH = "./data/netapp.json"
MEMGRAPH_URI = "bolt://127.0.0.1:7687"
MODEL = "llama3.2:latest"


def compute_tripplets_embeddings(driver, model):
    with driver.session() as session:
        # Retrieve all relationships
        result = session.run("MATCH (n:Chunk)-[r]->(m) RETURN n, r, m")
        print("Embedded data: ")

        for record in result:
            node1 = record["n"]
            relationship = record["r"]
            node2 = record["m"]
            # Check if the relationship already has an embedding
            if "embedding" in node1:
                print("Embedding already exists")
                return
            # Combine node labels and properties into a single string
            tripplet_data = (
                " ".join(node1.labels)
                + " "
                + " ".join(f"{k}: {v}" for k, v in node1.items())
                + " "
                + relationship.type
                + " "
                + " ".join(f"{k}: {v}" for k, v in relationship.items())
                + " "
                + " ".join(node2.labels)
                + " "
                + " ".join(f"{k}: {v}" for k, v in node2.items())
            )
            print(tripplet_data)
            # Compute the embedding for the tripplet
            tripplet_embedding = model.encode(tripplet_data)

            # Store the tripplet data on node1
            session.run(
                f"MATCH (n:Chunk) WHERE id(n) = {node1.element_id} SET n.embedding = {tripplet_embedding.tolist()}"
            )


def compute_node_embeddings(driver, model):
    print("compute_node_embeddings")
    with driver.session() as session:
        # Retrieve all nodes
        result = session.run("MATCH (n:Chunk) RETURN n")
        # print(f"Embedded data: {result.data()}")
        for record in result:
            node = record["n"]
            # Check if the node already has an embedding
            if "embedding" in node:
                print("Embedding already exists")
                return
            # print(node.element_id)
            # Combine node labels and properties into a single string
            node_data = (
                " ".join(node.labels)
                + " "
                + " ".join(f"{k}: {v}" for k, v in node.items())
            )
            # Compute the embedding for the node
            node_embedding = model.encode(node_data)
            ## print(node_embedding)
            # Store the embedding back into the node
            session.run(
                f"MATCH (n) WHERE id(n) = {node.element_id} SET n.embedding = {node_embedding.tolist()}"
            )
        # Set the label to Entity for all nodes
        session.run("MATCH (n) SET n:Entity")
        indexes = session.run("SHOW INDEX INFO")
        print(indexes.data())


def find_most_similar_node(driver, question_embedding):
    with driver.session() as session:
        # Perform the vector search on all nodes based on the question embedding
        result = session.run(
            f"CALL vector_search.search('chunk_index', 5, {question_embedding.tolist()}) YIELD * RETURN *;"
        )
        nodes_data = []

        # Retrieve all similar nodes and print them
        for record in result:
            node = record["node"]
            properties = {k: v for k, v in node.items() if k != "embedding"}
            node_data = {
                "distance": record["distance"],
                "id": node.element_id,
                "labels": list(node.labels),
                "properties": properties,
            }
            nodes_data.append(node_data)

        # Return the most similar node
        return nodes_data


def seed_database(driver, chunks):
    with driver.session() as session:
        # Clear the database
        session.run("MATCH (n) DETACH DELETE n")
        sleep(1)

        for chunk in chunks:
            session.run(merge_chunk_node_cypher, {"chunkParam": chunk})

        session.run(
            """CREATE VECTOR INDEX chunk_index ON :Chunk(embedding) WITH CONFIG {"dimension": 384, "capacity": 1000, "metric": "cos","resize_coefficient": 2}"""
        )


def count_nodes(driver):
    return driver.execute_query(count_nodes_cypher)


def show_indexes(driver):
    indexes = driver.execute_query("SHOW INDEX INFO")
    print(indexes)


# Not used
# async def get_response(client, prompt):
#     response = await client.chat.completions.create(
#         model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message.content


def get_llm_summary(context: str, question: str, short_answer: bool):
    template = short_template if short_answer else template
    p = ReviewDatasetPipeline(model=MODEL, template=template)
    return p.run(text=context, query=question)


def main(question, tripplets, short_answer):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = split_form10k_data_from_file(
        file=DATA_FILE_PATH,
        section_list=["item1", "item1a", "item7", "item7a"],
        text_splitter=text_splitter,
    )

    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("", ""))

    # Seed the database with some data
    seed_database(driver, chunks=chunks)
    count = count_nodes(driver)
    print(f"Node number = {count}")
    # Load the SentenceTransformer model
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    if tripplets:
        compute_tripplets_embeddings(driver, model)
    else:
        compute_node_embeddings(driver, model)

    question_embedding = model.encode(question)

    nodes = find_most_similar_node(driver, question_embedding)

    if len(nodes) > 0:
        sorted_nodes = sorted(nodes, key=lambda x: x["distance"], reverse=True)
        llm_response = get_llm_summary(
            context=sorted_nodes[0], question=question, short_answer=short_answer
        )
        print(llm_response[0])
    else:
        print("No matches found")


if __name__ == "__main__":
    question = "Tell me about the NetApp Company"

    main(question=question, tripplets=False, short_answer=True)
