from sentence_transformers import SentenceTransformer, util
import dotenv
dotenv.load_dotenv()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def find_closest(query, sentences):
    embeddings = model.encode(sentences)
    query_embedding = model.encode(query)

    closest_index = 0
    closest_score = -1

    for index, vector in enumerate(embeddings):
        similarity = util.cos_sim(vector, query_embedding).item()

        if similarity > closest_score:
            closest_score = similarity
            closest_index = index
            
    return sentences[closest_index]

def main():

    sentences = [
        "That is a sad person.",
        "That is a cute kitten.",
        "That is a beautiful place.",
        "Today is a rainy day."
        "That is a happy dog."
    ]

    query = "I love cats."

    embeddings = model.encode(sentences)
    query_embedding = model.encode(query)

    results = util.semantic_search(query_embedding, embeddings, top_k=2)
    print(results)


if __name__ == '__main__':
    main()