import logging
import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


ROOT_DIR = Path(__file__).resolve().parent
PDF_PATH = ROOT_DIR / "cisco-nexus-9000-series-nx-os-troubleshooting-guide-106x.pdf"
INDEX_PATH = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "cisco_nexus_troubleshooting"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "openai/gpt-oss-120b"
RETRIEVAL_K = 5

SYSTEM_PROMPT = """You are a technical assistant specializing in the Cisco Nexus 9000
Series Troubleshooting Guide.

Answer using only the retrieved document context as your primary and authoritative
source. Do not invent technical facts, commands, error messages, or procedures.
Clearly say when the context does not provide enough information. For conceptual
questions, synthesize relevant passages without claiming unsupported details are
in the guide. Be concise, structured, and technically clear. Cite the supplied
page labels when relevant.
"""

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
load_dotenv(ROOT_DIR / ".env")


def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_or_create_vectorstore(embeddings):
    database_file = INDEX_PATH / "chroma.sqlite3"
    vectorstore = None
    if database_file.is_file():
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(INDEX_PATH),
        )
        if vectorstore.get(include=["metadatas"])["ids"]:
            logger.info("Reusing the existing ChromaDB index at %s", INDEX_PATH)
            return vectorstore

    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"Troubleshooting PDF not found: {PDF_PATH}. "
            "Place the Cisco Nexus 9000 troubleshooting PDF beside app.py."
        )

    pages = PyPDFLoader(str(PDF_PATH)).load()
    if not any(page.page_content.strip() for page in pages):
        raise ValueError(
            f"No extractable text was found in {PDF_PATH.name}. "
            "Scanned PDFs require OCR, which this POC does not include."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=60,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError(f"No text chunks could be created from {PDF_PATH.name}.")

    logger.info("Creating the local vector index from %s", PDF_PATH.name)
    if vectorstore is None:
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(INDEX_PATH),
        )
    vectorstore.add_documents(chunks)
    return vectorstore


def create_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to the existing .env file "
            "in the project directory."
        )
    return ChatGroq(model=MODEL_NAME, temperature=0.1, api_key=api_key)


def answer_question(question, retriever, llm):
    if not question or not question.strip():
        raise ValueError("Enter a question before submitting.")

    documents = retriever.invoke(question.strip())
    if not documents:
        return "The supplied document context does not provide enough information to answer."

    context_parts = []
    pages = set()
    for document in documents:
        page = document.metadata.get("page")
        if isinstance(page, int):
            page_number = page + 1
            pages.add(page_number)
            context_parts.append(f"[Page {page_number}]\n{document.page_content}")
        else:
            context_parts.append(document.page_content)

    prompt = (
        f"{SYSTEM_PROMPT}\n"
        f"Retrieved document context:\n\n{'\n\n'.join(context_parts)}\n\n"
        f"User question:\n{question.strip()}\n\n"
        "Answer:"
    )
    response = llm.invoke(prompt)
    answer = str(response.content).strip()
    if pages:
        page_list = ", ".join(str(page) for page in sorted(pages))
        answer += (
            "\n\nSource: Cisco Nexus 9000 Troubleshooting Guide, "
            f"page{'s' if len(pages) > 1 else ''} {page_list}."
        )
    return answer


def format_conversation_history(entries):
    if not entries:
        return "No questions yet."
    return "\n\n---\n\n".join(
        f"**Q:** {entry['question']}\n\n**A:** {entry['answer']}"
        for entry in reversed(entries)
    )


def handle_question(question, chat_messages, conversation_history, retriever, llm):
    chat_messages = chat_messages or []
    conversation_history = conversation_history or []
    if not question or not question.strip():
        return (
            question or "",
            chat_messages,
            conversation_history,
            format_conversation_history(conversation_history),
            "Please enter a question.",
        )

    try:
        answer = answer_question(question, retriever, llm)
    except Exception as error:
        logger.exception("Question answering failed (%s)", type(error).__name__)
        return (
            question,
            chat_messages,
            conversation_history,
            format_conversation_history(conversation_history),
            "Unable to answer this question. Check the app terminal, the Groq API "
            "configuration, network connection, and model availability.",
        )

    chat_messages = chat_messages + [
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": answer},
    ]
    conversation_history = conversation_history + [
        {"question": question.strip(), "answer": answer}
    ]
    return (
        "",
        chat_messages,
        conversation_history,
        format_conversation_history(conversation_history),
        "",
    )


def build_interface(retriever, llm):
    with gr.Blocks(title="Cisco Nexus 9000 Troubleshooting Assistant") as demo:
        gr.Markdown(
            "# Cisco Nexus 9000 Troubleshooting Assistant\n"
            "Answers are generated from the supplied Cisco Nexus 9000 "
            "troubleshooting guide."
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("## Conversation History")
                history_view = gr.Markdown("No questions yet.")
                conversation_history = gr.State([])
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Troubleshooting Guide",
                    height=520,
                )
                with gr.Row():
                    question = gr.Textbox(
                        label="Question",
                        placeholder="Ask a question about the guide...",
                        lines=2,
                        scale=5,
                    )
                    ask_button = gr.Button("Ask", variant="primary", scale=1)
                status = gr.Markdown()
                clear_button = gr.Button("Clear Chat")

        outputs = [question, chatbot, conversation_history, history_view, status]

        def submit_question(text, messages, entries):
            return handle_question(text, messages, entries, retriever, llm)

        ask_button.click(
            submit_question,
            inputs=[question, chatbot, conversation_history],
            outputs=outputs,
        )
        question.submit(
            submit_question,
            inputs=[question, chatbot, conversation_history],
            outputs=outputs,
        )
        clear_button.click(
            lambda: ("", [], [], "No questions yet.", ""),
            outputs=outputs,
        )
    return demo


def main():
    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"Troubleshooting PDF not found: {PDF_PATH}. "
            "Place the Cisco Nexus 9000 troubleshooting PDF beside app.py."
        )
    llm = create_llm()
    embeddings = create_embeddings()
    vectorstore = load_or_create_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K},
    )
    demo = build_interface(retriever, llm)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,
        show_error=False,
    )


if __name__ == "__main__":
    main()
