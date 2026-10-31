from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.documents import Document

from app import (
    RETRIEVAL_K,
    answer_question,
    format_conversation_history,
    handle_question,
)


def test_answer_question_uses_retrieved_context_and_cites_pages():
    retriever = Mock()
    retriever.invoke.return_value = [
        Document(page_content="The switch reports a fan fault.", metadata={"page": 11}),
        Document(page_content="Inspect the fan module.", metadata={"page": 12}),
    ]
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(content="The guide reports a fan fault.")

    answer = answer_question("What does the fault indicate?", retriever, llm)

    retriever.invoke.assert_called_once_with("What does the fault indicate?")
    prompt = llm.invoke.call_args.args[0]
    assert "[Page 12]" in prompt
    assert "[Page 13]" in prompt
    assert "The switch reports a fan fault." in prompt
    assert "pages 12, 13" in answer
    assert RETRIEVAL_K == 5


def test_answer_question_rejects_empty_questions_without_retrieval():
    retriever = Mock()
    llm = Mock()

    try:
        answer_question("   ", retriever, llm)
    except ValueError as error:
        assert str(error) == "Enter a question before submitting."
    else:
        raise AssertionError("A blank question must be rejected.")

    retriever.invoke.assert_not_called()
    llm.invoke.assert_not_called()


def test_handle_question_updates_chat_and_session_history():
    retriever = Mock()
    retriever.invoke.return_value = [
        Document(page_content="Use the documented troubleshooting procedure.", metadata={})
    ]
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(content="Follow the documented procedure.")

    cleared_input, chat, entries, history_view, status = handle_question(
        "How should I troubleshoot this?", [], [], retriever, llm
    )

    assert cleared_input == ""
    assert chat == [
        {"role": "user", "content": "How should I troubleshoot this?"},
        {"role": "assistant", "content": "Follow the documented procedure."},
    ]
    assert entries == [
        {
            "question": "How should I troubleshoot this?",
            "answer": "Follow the documented procedure.",
        }
    ]
    assert "**Q:** How should I troubleshoot this?" in history_view
    assert status == ""


def test_handle_question_reports_empty_input_without_changing_history():
    retriever = Mock()
    llm = Mock()

    result = handle_question(" ", [], [], retriever, llm)

    assert result[1] == []
    assert result[2] == []
    assert result[4] == "Please enter a question."
    retriever.invoke.assert_not_called()
    llm.invoke.assert_not_called()


def test_history_is_newest_first_and_empty_state_is_clear():
    assert format_conversation_history([]) == "No questions yet."
    formatted = format_conversation_history(
        [
            {"question": "First question", "answer": "First answer"},
            {"question": "Second question", "answer": "Second answer"},
        ]
    )
    assert formatted.index("Second question") < formatted.index("First question")
