from groq import Groq
from app.config import GROQ_API_KEY, LLM_MODEL

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a precise document Q&A assistant.
Answer ONLY using the provided context chunks.
If the answer is not in the context, say:
'I don\'t have enough information in the provided documents.'
Be concise and factual."""

def generate_answer(query: str, context_docs: list) -> dict:
    if not context_docs:
        return {
            "answer": "No documents ingested yet. Please upload a PDF first.",
            "source_chunks": []
        }

    context  = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.1,
        max_tokens=512
    )

    answer  = response.choices[0].message.content
    sources = [doc.page_content[:200] + "..." for doc in context_docs]
    print(f"[Generator] Answer: {len(answer)} chars")
    return {"answer": answer, "source_chunks": sources}
