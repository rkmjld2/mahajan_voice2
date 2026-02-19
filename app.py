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
from groq import Groq  # Official Groq client for Whisper transcription

st.title("📄 RAG PDF Voice Assistant (Groq)")
st.info("Upload PDF → Ask via text or 🎤 voice → Get text + spoken answer.")

# ────────────────────────────────────────────────
# Secrets & Clients
# ────────────────────────────────────────────────
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Session state for vectorstore (persist across reruns)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()

    if len(pdf_bytes) < 200:
        st.error("Uploaded file is empty or too small.")
    else:
        try:
            with st.spinner("Loading PDF from memory..."):
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                docs_list = []
                for page_num in range(len(doc)):
                    text = doc[page_num].get_text("text")
                    metadata = {"source": uploaded_file.name or "uploaded.pdf", "page": page_num + 1}
                    docs_list.append(Document(page_content=text, metadata=metadata))
                doc.close()

                if not docs_list or all(len(d.page_content.strip()) == 0 for d in docs_list):
                    st.warning("No readable text extracted (possibly scanned/image-only PDF).")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
                    splits = text_splitter.split_documents(docs_list)

                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vectorstore = FAISS.from_documents(splits, embeddings)
                    st.session_state.vectorstore = vectorstore
                    st.success(f"PDF ready! {len(docs_list)} pages processed.")

        except Exception as e:
            st.error(f"PDF processing failed: {str(e)}")

# ────────────────────────────────────────────────
# Query section — only if PDF loaded
# ────────────────────────────────────────────────
if st.session_state.vectorstore is not None:

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer based only on the provided context. Be concise and accurate.

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

    # ── Input: Text or Voice (microphone) ──
    user_input = st.chat_input(
        placeholder="Ask about the PDF (type or click mic 🎤 to speak)...",
        accept_audio=True
    )

    if user_input:

        question = ""

        if isinstance(user_input, str):
            # Plain text input
            question = user_input.strip()

        elif isinstance(user_input, dict) and "audio" in user_input:
            # Voice input recorded
            audio_upload = user_input["audio"]  # Streamlit UploadedFile-like object
            if audio_upload:
                audio_bytes = audio_upload.read()

                with st.spinner("Transcribing your voice using Groq Whisper..."):
                    try:
                        # Use Groq's OpenAI-compatible Whisper endpoint
                        transcription = groq_client.audio.transcriptions.create(
                            file=("audio.wav", audio_bytes, "audio/wav"),
                            model="whisper-large-v3-turbo",  # fast & good balance (or "whisper-large-v3")
                            response_format="text",
                            language="en"  # optional: force English; remove for auto-detect
                        )
                        question = transcription.strip()
                        st.caption(f"**You said:** {question}")

                    except Exception as trans_e:
                        st.error(f"Voice transcription failed: {str(trans_e)}")
                        question = ""

        if question:
            with st.spinner("Generating answer..."):
                try:
                    response = rag_chain.invoke(question)
                    answer_text = response.content.strip()

                    st.subheader("Answer")
                    st.markdown(answer_text)

                    # Spoken output
                    with st.spinner("Preparing voice answer..."):
                        tts = gTTS(text=answer_text, lang='en')
                        audio_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                        tts.save(audio_tmp.name)
                        audio_path = audio_tmp.name

                        st.audio(audio_path, format="audio/mp3")

                        with open(audio_path, "rb") as af:
                            st.download_button(
                                label="Download spoken answer (MP3)",
                                data=af,
                                file_name="answer.mp3",
                                mime="audio/mp3"
                            )

                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

            # Cleanup
            if 'audio_path' in locals() and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except:
                    pass

else:
    st.info("Upload a PDF first to enable questions (text or voice).")