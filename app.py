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

st.title("📄 RAG PDF Voice Assistant (Groq)")
st.info("Upload PDF → Ask via text or 🎤 voice → Get text + spoken answer.")

# Clients & Session
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    st.write(f"PDF uploaded - size: {len(pdf_bytes)} bytes")

    if len(pdf_bytes) < 200:
        st.error("File too small.")
    else:
        try:
            with st.spinner("Loading PDF..."):
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                docs_list = []
                for page_num in range(len(doc)):
                    text = doc[page_num].get_text("text")
                    docs_list.append(Document(page_content=text, metadata={"page": page_num + 1}))
                doc.close()

                if not docs_list:
                    st.warning("No text extracted.")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
                    splits = text_splitter.split_documents(docs_list)

                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vectorstore = FAISS.from_documents(splits, embeddings)
                    st.session_state.vectorstore = vectorstore
                    st.success(f"PDF ready! {len(docs_list)} pages.")
        except Exception as e:
            st.error(f"PDF load error: {str(e)}")

# ── If PDF loaded ──
if st.session_state.vectorstore is not None:
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer based only on the context. Be concise.

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

    # ── Input ──
    user_input = st.chat_input(
        "Ask about the PDF (type or mic)...",
        accept_audio=True
    )

    if user_input:
        st.write("Input received!")  # Debug: confirm input detected

        question = ""

        if isinstance(user_input, str):
            question = user_input.strip()
            st.write(f"Text input: '{question}'")

        elif isinstance(user_input, dict) and "audio" in user_input:
            audio_upload = user_input.get("audio")
            if audio_upload:
                audio_bytes = audio_upload.read()
                st.write(f"Audio received - size: {len(audio_bytes)} bytes")

                with st.spinner("Transcribing voice (Groq Whisper)..."):
                    try:
                        # Correct file= format: tuple (filename, bytes, mime)
                        transcription_resp = groq_client.audio.transcriptions.create(
                            file=("audio_from_streamlit.wav", audio_bytes, "audio/wav"),
                            model="whisper-large-v3-turbo",
                            response_format="text",
                            language="en"
                        )
                        question = transcription_resp.strip() if transcription_resp else ""
                        st.caption(f"You said (transcribed): **{question}**")

                        if not question:
                            st.warning("Transcription returned empty string.")
                    except Exception as trans_err:
                        st.error(f"Transcription failed: {str(trans_err)}")
                        question = ""

        if question:
            st.write(f"Processing question: '{question}'")
            with st.spinner("Generating answer..."):
                try:
                    response = rag_chain.invoke(question)
                    answer_text = response.content.strip()
                    st.subheader("Answer")
                    st.markdown(answer_text)

                    with st.spinner("Voice answer..."):
                        tts = gTTS(text=answer_text, lang='en')
                        audio_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                        tts.save(audio_tmp.name)
                        audio_path = audio_tmp.name

                        st.audio(audio_path, format="audio/mp3")

                        with open(audio_path, "rb") as af:
                            st.download_button("Download MP3", af, "answer.mp3", "audio/mp3")

                except Exception as gen_err:
                    st.error(f"Answer generation failed: {str(gen_err)}")
        else:
            st.info("No valid question after processing (empty transcription or input). Try again.")

else:
    st.info("Upload PDF first.")
