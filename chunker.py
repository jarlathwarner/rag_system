from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_fixed(text: str, max_chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks based on a fixed character length,
    respecting paragraph boundaries where possible.
    """
    if not text or len(text.strip()) == 0:
        return []

    paragraphs = text.split("\n\n")
    chunks = []
    for i in range(0, len(paragraphs), max_chunk_size - overlap):
        chunk = ' '.join(paragraphs[i:i + max_chunk_size])
        chunks.append(chunk)

    return chunks

def semantic_chunking(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)