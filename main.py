from sentence_transformers import SentenceTransformer
from numpy import dot , linalg



model = SentenceTransformer("all-MiniLM-L6-v2") 

N = 500
text =""


def chunk_text(text : str , chunk_size : int ) -> list[str]:
    chunks = []
    current_chunk= ""

    sentences = text.split(". ")

    for sentence in sentences :
        if len(current_chunk + sentence) < chunk_size :
            current_chunk += sentence + ". "
        else :
            chunks.append(current_chunk) 
            current_chunk = sentence
    if current_chunk :
        chunks.append(current_chunk)
        
    return chunks

def cosim_similarity(chunkonevec , chunktwovec) :
    return dot(chunkonevec , chunktwovec)/(linalg.norm(chunkonevec) * linalg.norm(chunktwovec))

with open("data/data.txt", "r", encoding="utf-8") as file:
    text = file.read()

chunks = chunk_text(text,N)
firstvec = model.encode(chunks[0])
secondvec = model.encode(chunks[1])

results = cosim_similarity(firstvec,firstvec)
print(results)