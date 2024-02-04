from sentence_transformers import SentenceTransformer, util

query = "How many people live in London?"
docs = ["Around 9 Million people live in London", "London is known for its financial district"]

#Load the model

def get_similarity(query, compareSents):
#Encode query and documents
    model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    query_emb = model.encode(query)
    doc_emb = model.encode(compareSents)

    #Compute dot score between query and all document embeddings
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    #Combine docs & scores
    doc_score_pairs = list(zip(compareSents, scores))

    #Sort by decreasing score
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    return doc_score_pairs
    #Output passages & scores
    for doc, score in doc_score_pairs:
        print(score, doc)
print(get_similarity(query,docs))