

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

with open("data/data.txt", "r", encoding="utf-8") as file:
    text = file.read()

chunks = chunk_text(text,N)
print(len(chunks))
print(chunks[0])
print(chunks[-1])
