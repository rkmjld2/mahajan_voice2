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

# Page setup
st.set_page_config(page_title="RAG PDF Voice Assistant", layout="wide")

st.title("📄 RAG PDF Voice Assistant (Groq)")
st.info("Upload PDF → Ask via text or microphone → Get text + spoken answer")

# Groq client
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Session state for vectorstore
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# PDF upload
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    st.write(f"PDF uploaded - size: {len(pdf_bytes):,} bytes")

    if len(pdf_bytes) < 200:
        st.error("File too small or empty.")
    else:
        try:
            with st.spinner("Loading PDF..."):
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
                    st.warning("No readable text extracted (maybe scanned PDF?).")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
                    splits = text_splitter.split_documents(docs_list)

                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vectorstore = FAISS.from_documents(splits, embeddings)
                    st.session_state.vectorstore = vectorstore
                    st.success(f"PDF ready! ({len(docs_list)} pages)")
        except Exception as e:
            st.error(f"PDF processing failed: {str(e)}")

# ────────────────────────────────────────────────
# Chat section
# ────────────────────────────────────────────────
if st.session_state.vectorstore is not None:

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer based only on the context. Be concise and accurate.

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

    # ── Input with text + voice ──
    user_input = st.chat_input(
        placeholder="Ask about the PDF (type or speak)...",
        accept_audio=True
    )

    if user_input is not None:
        st.write("Input received!")

        question = ""

        # ── Correct access for ChatInputValue object ──
        if hasattr(user_input, "text") and user_input.text:
            question = user_input.text.strip()
            st.write(f"**Typed:** {question}")

        if hasattr(user_input, "audio") and user_input.audio:
            audio_upload = user_input.audio
            audio_bytes = audio_upload.getvalue()
            st.write(f"Audio captured - size: {len(audio_bytes):,} bytes")

            with st.spinner("Transcribing voice..."):
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
                        st.warning("Transcription came back empty.")
                except Exception as e:
                    st.error(f"Transcription failed: {str(e)}")

        if not question:
            st.info("No question detected. Try typing clearly or speaking louder and longer (3-6 seconds).")

        if question:
            with st.spinner("Generating answer..."):
                try:
                    response = rag_chain.invoke(question)
                    answer_text = response.content.strip()

                    st.subheader("Answer")
                    st.markdown(answer_text)

                    with st.spinner("Creating voice..."):
                        tts = gTTS(text=answer_text, lang='en')
                        tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                        tts.save(tmp_audio.name)

                        st.audio(tmp_audio.name, format="audio/mp3")

                        with open(tmp_audio.name, "rb") as f:
                            st.download_button(
                                label="Download MP3",
                                data=f,
                                file_name="answer.mp3",
                                mime="audio/mp3"
                            )

                except Exception as e:
                    st.error(f"Answer generation failed: {str(e)}")

            # Cleanup
            if 'tmp_audio' in locals():
                try:
                    os.unlink(tmp_audio.name)
                except:
                    pass

else:
    st.info("Upload a PDF first to ask questions.")
