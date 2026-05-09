import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# we use text-embedding-3-small for cost-effective and efficient embedding generation;
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# defining bot personas to represent different perspectives: tech-optimist, skeptic, and finance-focused;
BOT_PERSONAS = {
    "bot_a": "I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns.",
    "bot_b": "I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical of AI, social media, and billionaires. I value privacy and nature.",
    "bot_c": "I strictly care about markets, interest rates, trading algorithms, and making money. I speak in finance jargon and view everything through the lens of ROI."
}

def _init_vector_store():
    """Initializes a FAISS vector store with bot personas for retrieval."""
    ids = list(BOT_PERSONAS.keys())
    texts = list(BOT_PERSONAS.values())
    
    vector_store = FAISS.from_texts(
        texts, 
        embeddings, 
        metadatas=[{"bot_id": id} for id in ids]
    )
    return vector_store

# persistent vector store instance for routing operations;
vector_store = _init_vector_store()

def route_post_to_bots(post_content: str, threshold: float = 0.3) -> list[dict]:
    """
    Identifies bots whose personas align with the post content based on cosine similarity.
    Returns a sorted list of matches exceeding the similarity threshold.
    """
    post_embedding = np.array(embeddings.embed_query(post_content))
    matched_bots = []
    
    for bot_id, persona in BOT_PERSONAS.items():
        # re-embedding personas locally for simplicity in this demo; 
        # for high-scale use, pre-calculate and store these embeddings;
        bot_embedding = np.array(embeddings.embed_query(persona))
        
        # calculate manual cosine similarity: (a dot b) / (||a|| * ||b||);
        dot_product = np.dot(post_embedding, bot_embedding)
        norm_post = np.linalg.norm(post_embedding)
        norm_bot = np.linalg.norm(bot_embedding)
        similarity = dot_product / (norm_post * norm_bot)
        
        if similarity > threshold:
            matched_bots.append({
                "bot_id": bot_id,
                "persona": persona,
                "similarity": float(similarity)
            })
            
    return sorted(matched_bots, key=lambda x: x["similarity"], reverse=True)

if __name__ == "__main__":
    test_post = "OpenAI just released a new model that might replace junior developers."
    print(f"Routing post: \"{test_post}\"\n")
    
    results = route_post_to_bots(test_post)
    
    if not results:
        print("No bots matched the similarity threshold.")
    else:
        for match in results:
            print(f"Match: {match['bot_id']} (Score: {match['similarity']:.4f})")
            print(f"Persona: {match['persona']}\n")
