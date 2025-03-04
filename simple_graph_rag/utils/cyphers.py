merge_chunk_node_cypher = """
    MERGE(mergedChunk:Chunk {chunkId: $chunkParam.chunkId})
        ON CREATE SET 
            mergedChunk.names = $chunkParam.names,
            mergedChunk.formId = $chunkParam.formId, 
            mergedChunk.cik = $chunkParam.cik, 
            mergedChunk.cusip6 = $chunkParam.cusip6, 
            mergedChunk.source = $chunkParam.source, 
            mergedChunk.f10kItem = $chunkParam.f10kItem, 
            mergedChunk.chunkSeqId = $chunkParam.chunkSeqId, 
            mergedChunk.text = $chunkParam.text
    RETURN mergedChunk
"""
unique_constraint_cypher = """
    CREATE CONSTRAINT ON (c:Chunk) ASSERT c.chunkId IS UNIQUE
"""
# empty database
delete_all_cypher = """
    MATCH (n)
    DETACH DELETE n
"""

# Drop graph
# You must be in STORAGE MODE IN_MEMORY_ANALYTICAL
# The dfualt seems to be IN_MEMORY_TRANSACTIONAL"
drop_graph = """
    DROP GRAPH;
"""
count_nodes_cypher = """
    MATCH (n)
    RETURN count(n) as nodeCount
"""
show_index_info = """SHOW INDEX INFO"""

create_text_embedding_index = """
    CREATE INDEX ON :Chunk(textEmbedding);
"""

create_vector_index = """
    CREATE VECTOR INDEX chunk_vector_index ON :Chunk(textEmbedding) WITH CONFIG {"dimension": 384, "capacity": 1000, "metric": "cos","resize_coefficient": 2}
"""
show_vector_indices = "CALL vector_search.show_index_info() YIELD * RETURN *;"
search_cypher = "CALL vector_search.search({index_name}, {limit}, {embedded_question}) YIELD * RETURN *;"
