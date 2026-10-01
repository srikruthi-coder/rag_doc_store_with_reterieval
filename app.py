import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

st.set_page_config(page_title="Mini RAG", page_icon="📚")

st.title("Mini RAG: Document Store + Retrieval")

st.caption("PDF → Chunks → Embeddings → ChromaDB → Retrieval → Ollama")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="documents")


# Sidebar settings
st.sidebar.header("Settings")

ollama_model = st.sidebar.text_input(
    "Ollama model",
    "llama3.2"
)

chunk_size = st.sidebar.slider(
    "Chunk size",
    200,
    1500,
    500,
    100
)

top_k = st.sidebar.slider(
    "Chunks to retrieve",
    1,
    5,
    3
)


# Build Document Store
st.header("1. Build Document Store")

uploaded_file = st.file_uploader(
    "Upload a text-based PDF",
    type=["pdf"]
)


if uploaded_file and st.button("Process & Store PDF"):

    reader = PdfReader(uploaded_file)

    text = ""

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    if not text.strip():
        st.error("No readable text was found. Try a text-based PDF.")
        st.stop()

    # Create chunks
    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

    st.write(f"Created {len(chunks)} chunks.")

    # Generate embeddings
    with st.spinner("Generating embeddings..."):

        embeddings = model.encode(chunks)

    # Delete existing documents
    existing = collection.get()

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # Store chunks and embeddings
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist()
    )

    st.success(
        f"Successfully stored {len(chunks)} chunks in ChromaDB."
    )


# Ask Questions
st.header("2. Ask Questions")

question = st.text_input(
    "Ask a question about your PDF"
)

if st.button("Ask AI"):

    if not question.strip():

        st.warning("Please enter a question.")

    elif collection.count() == 0:

        st.warning("Please process a PDF first.")

    else:

        # Convert question into embedding
        question_embedding = model.encode([question])[0]

        # Retrieve relevant chunks
        results = collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=top_k
        )

        retrieved_chunks = results["documents"][0]

        # Combine retrieved chunks
        context = "\n\n".join(retrieved_chunks)

        # Prompt LLM
        prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question-answering ONLY using the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context,
say "I don't know based on the provided document."
"""

        # Send to Ollama
        with st.spinner("Generating answer..."):

            try:

                response = ollama.chat(
                    model=ollama_model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                st.subheader("Answer")

                st.write(
                    response["message"]["content"]
                )

                # Show retrieved chunks
                with st.expander("Retrieved Context"):

                    for i, chunk in enumerate(
                        retrieved_chunks,
                        start=1
                    ):

                        st.write(f"**Chunk {i}:**")
                        st.write(chunk)

            except Exception as e:

                st.error(
                    "Could not connect to Ollama. "
                    "Make sure Ollama is running and "
                    "the selected model is installed."
                )

                st.code(str(e))


st.divider()

st.caption(
    "PDF → Chunks → Embeddings → ChromaDB → Retrieval → Ollama"
)