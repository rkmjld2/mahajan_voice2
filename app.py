import streamlit as st
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from gtts import gTTS
import tempfile
import os
from groq import Groq

# ────────────────────────────────────────────────
# Page config & title
# ────────────────────────────────────────────────
st.set_page_config(page_title="RAG PDF Voice Assistant", layout="wide")

st.title("📄 RAG PDF Voice Assistant (Groq)")
st.info("Upload PDF → Ask questions via text or microphone → Get text + spoken answer")

# ────────────────────────────────────────────────
# Initialize Groq client
# ────────────────────────────────────────────────
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Session state for vector store (so we don't re-process PDF every time)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ────────────────────────────────────────────────
# PDF Upload & Processing
# ────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    st.write(f"PDF uploaded - size: {len(pdf_bytes):,} bytes")

    if len(pdf_bytes) < 200:
        st.error("Uploaded file is too small or empty.")
    else:
        try:
            with st.spinner("Loading and indexing PDF..."):
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                docs_list = []

                for page_num in range(len(doc)):
                    text = doc[page_num].get_text("text")
                    docs_list.append(
                        Document(
                            page_content=text,
                            metadata={"source": uploaded_file.name, "page": page_num + 1}
                        )
                    )

                doc.close()

                if not docs_list or all(len(d.page_content.strip()) == 0 for d in docs_list):
                    st.warning("No readable text could be extracted from this PDF.")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=120
                    )
                    splits = text_splitter.split_documents(docs_list)

                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    vectorstore = FAISS.from_documents(splits, embeddings)
                    st.session_state.vectorstore = vectorstore

                    st.success(f"PDF processed successfully! ({len(docs_list)} pages)")
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")

# ────────────────────────────────────────────────
# Question & Answer section (only if PDF is loaded)
# ────────────────────────────────────────────────
if st.session_state.vectorstore is not None:

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the question based **only** on the following context.
Be concise, accurate and polite.

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    # ── Chat input with voice support ──
    user_input = st.chat_input(
        placeholder="Ask about the PDF (type or speak with microphone)...",
        accept_audio=True
    )

    if user_input:

        question = ""

        # ── Handle different possible return types ──
        if isinstance(user_input, dict):
            st.write("Input received (dict format)")

            # Case 1: User typed text
            if "text" in user_input and user_input["text"]:
                question = user_input["text"].strip()
                st.write(f"**Typed question:** {question}")

            # Case 2: User recorded voice
            elif "audio" in user_input and user_input["audio"]:
                audio_file = user_input["audio"]
                audio_bytes = audio_file.getvalue()

                st.write(f"Audio received — size: {len(audio_bytes):,} bytes")

                with st.spinner("Transcribing your voice (Groq Whisper)..."):
                    try:
                        transcription = groq_client.audio.transcriptions.create(
                            file=("audio.wav", audio_bytes, "audio/wav"),
                            model="whisper-large-v3-turbo",
                            response_format="text",
                            language="en"
                        )

                        question = (transcription or "").strip()

                        if question:
                            st.caption(f"**You said:** {question}")
                        else:
                            st.warning("Transcription returned empty result.")
                    except Exception as e:
                        st.error(f"Voice transcription failed: {str(e)}")

            else:
                st.warning("Input dict has no 'text' or 'audio' key.")

        elif isinstance(user_input, str):
            # Rare fallback case
            question = user_input.strip()
            st.write(f"**Typed question (string):** {question}")

        else:
            st.warning(f"Unexpected input type: {type(user_input)}")

        # ── If we have a question → generate answer ──
        if question:
            with st.spinner("Thinking..."):
                try:
                    response = rag_chain.invoke(question)
                    answer_text = response.content.strip()

                    st.subheader("Answer")
                    st.markdown(answer_text)

                    # Text-to-speech output
                    with st.spinner("Preparing spoken answer..."):
                        tts = gTTS(text=answer_text, lang='en')
                        tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                        tts.save(tmp_audio.name)

                        st.audio(tmp_audio.name, format="audio/mp3")

                        # Download button
                        with open(tmp_audio.name, "rb") as f:
                            st.download_button(
                                label="Download spoken answer",
                                data=f,
                                file_name="answer.mp3",
                                mime="audio/mp3"
                            )

                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

            # Cleanup temporary audio file
            if 'tmp_audio' in locals() and os.path.exists(tmp_audio.name):
                try:
                    os.unlink(tmp_audio.name)
                except:
                    pass

        else:
            st.info("No question could be extracted. Please try typing or speaking again.")

else:
    st.info("Please upload a PDF file first to start asking questions.")
