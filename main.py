from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

import os

# ✅ Load environment variables
load_dotenv()

# ✅ Initialize Mistral client
client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

# ✅ Embedding model (FREE)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ✅ Load FAISS vector database
vectorstore = FAISS.load_local(
    "faiss_db",
    embedding_model,
    allow_dangerous_deserialization=True
)

# ✅ Create retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# ✅ Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("RAG system created ✅")
print("Press 0 to exit")

# ✅ Chat loop
while True:
    query = input("You: ")

    if query == "0":
        break

    # 🔍 Retrieve relevant chunks
    # docs = retriever.invoke(query)

    # context = "\n\n".join(
    #     [doc.page_content for doc in docs]
    # )

    # 🔍 Retrieve relevant chunks WITH scores
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=4)

    context = ""
    sources = []

    for doc, score in docs_with_scores:
        context += doc.page_content + "\n\n"

        sources.append({
            "content": doc.page_content[:200],  # short preview
            "score": round(score, 4),
            "page": doc.metadata.get("page", "N/A")
        })
    
    confidence = round(
        sum([s["score"] for s in sources]) / len(sources),4
    )

    # 🧠 Build prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # 🤖 Call Mistral
    response = client.chat(
        model="mistral-small",
        messages=[
            ChatMessage(
                role="user",
                content=final_prompt.to_string()
            )
        ]
    )

    # 📊 Print confidence
    print("📊 Confidence Score:", confidence)

    # 🖨️ Print response
    print(f"\nAI: {response.choices[0].message.content}\n")


    # 📚 Print sources
    print("\n📚 Sources Used:\n")
    for i, src in enumerate(sources, 1):
        print(f"{i}. Page: {src['page']} (score: {src['score']})")
        print(src["content"])
        print("-" * 50)