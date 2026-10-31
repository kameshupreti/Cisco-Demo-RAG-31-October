import sys

from app import create_embeddings, create_llm, load_or_create_vectorstore, answer_question


FACTUAL_QUESTION = (
    "According to the guide's initial port verification checklist, what four "
    "checks should an engineer perform to verify a port is connected correctly "
    "and operational?"
)
CONCEPTUAL_QUESTION = (
    "Based on the Cisco Nexus 9000 troubleshooting guide, describe a sensible "
    "step-by-step approach an engineer can follow to investigate a recurring "
    "network or switch issue. Explain how the engineer can use the documented "
    "symptoms and troubleshooting procedures to narrow down the possible cause."
)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    llm = create_llm()
    embeddings = create_embeddings()
    vectorstore = load_or_create_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    failed_tests = []
    for label, question in (
        ("Direct factual lookup", FACTUAL_QUESTION),
        ("Broader conceptual question", CONCEPTUAL_QUESTION),
    ):
        print(f"\n--- {label} ---\nQuestion: {question}")
        try:
            answer = answer_question(question, retriever, llm)
        except Exception as error:
            print(f"BLOCKED: {type(error).__name__}; check the Groq API configuration.")
            failed_tests.append(label)
        else:
            print(f"PASS\nAnswer:\n{answer}\n")

    if failed_tests:
        print(f"Failed or blocked smoke tests: {', '.join(failed_tests)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
