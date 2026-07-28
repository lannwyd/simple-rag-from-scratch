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

def retrieve(query : str , chunks : list[str] , model , top_k = 3) :
    question_vector = model.encode(query)
    i = 0
    numbers = []
    for chunk in chunks :
        vector = model.encode(chunk)
        numbers.append((cosim_similarity(question_vector , vector) , chunk))
    numbers.sort(reverse=True)
    results = numbers[0:top_k]
    return results

def cosim_similarity(chunkonevec , chunktwovec) :
    return dot(chunkonevec , chunktwovec)/(linalg.norm(chunkonevec) * linalg.norm(chunktwovec))

with open("data/data.txt", "r", encoding="utf-8") as file:
    text = file.read()

chunks = chunk_text(text,N)

qst = input("ask your question :\n")

top = retrieve(qst , chunks ,model)
print(top)