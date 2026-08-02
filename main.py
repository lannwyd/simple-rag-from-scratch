from sentence_transformers import SentenceTransformer
from numpy import dot , linalg
import os
from dotenv import load_dotenv 
from groq import Groq

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
client = Groq()

model = SentenceTransformer("all-MiniLM-L6-v2") 

N = 500
text =""

with open("data/data.txt", "r", encoding="utf-8") as file:
    text = file.read()

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
    results = numbers[0:top_k ]
    return results

def cosim_similarity(chunkonevec , chunktwovec) :
    return dot(chunkonevec , chunktwovec)/(linalg.norm(chunkonevec) * linalg.norm(chunktwovec))

def generate_answer(question , context) :
    client = Groq()

    stream = client.chat.completions.create(
        messages=[
            
            {
                "role": "system",
                "content": "Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.\n\n context:\n" + context,
            },
            {
                "role": "user",
                "content": question
            }
            
        ],
        model="openai/gpt-oss-120b",

        temperature=0.5,

        max_completion_tokens=1024,

        top_p=1,

        stop=None,

        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content is not None:
                print(chunk.choices[0].delta.content, end="")

chunks = chunk_text(text,N)
    
qst = input("ask your question :\n")
top = retrieve(qst , chunks ,model)

cominedchunks = ""
for number , chunk in top:
    cominedchunks += chunk

generate_answer(qst , cominedchunks)