from utils.splitter_utils import Embedder


def answer_question(client, question):
    nodes_data = []
    embedded_question = Embedder.embed_text(question)
    with client.session() as session:
        result = session.run(
            f"CALL vector_search.search('got_index', 10, {embedded_question.tolist()}) YIELD * RETURN *;"
        )

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
        print("All similar nodes:")
        for node in nodes_data:
            print(node)

        # Return the most similar node
        return nodes_data[0] if nodes_data else None
