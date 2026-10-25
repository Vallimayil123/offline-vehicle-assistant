
from unittest.mock import Mock, patch

import pytest

from src.assistant import VehicleAssistant


def test_answer_uses_retrieved_manual_passages():
    ingestor = Mock()
    ingestor.search_manual.return_value = [
        {
            "text": "Check the engine oil level regularly.",
            "page": 3,
            "source": "sample_manual.pdf",
            "distance": 0.1,
        }
    ]

    assistant = VehicleAssistant(ingestor=ingestor)

    with patch("src.assistant.ollama.chat") as mock_chat:
        mock_chat.return_value = {
            "message": {
                "content": "Check the oil level regularly (sample_manual.pdf, page 3)."
            }
        }

        response = assistant.answer("How do I check engine oil?")

    assert "answer" in response
    assert "page 3" in response["answer"]
    assert len(response["sources"]) == 1
    assert response["sources"][0]["page"] == 3
    ingestor.search_manual.assert_called_once_with(
        "How do I check engine oil?",
        n_results=3,
    )


def test_blank_question_is_rejected():
    assistant = VehicleAssistant(ingestor=Mock())

    with pytest.raises(ValueError):
        assistant.answer("   ")


def test_no_relevant_passages_returns_no_sources():
    ingestor = Mock()
    ingestor.search_manual.return_value = []

    assistant = VehicleAssistant(ingestor=ingestor)
    response = assistant.answer("What is my car's paint color?")

    assert response["sources"] == []
    assert "couldn't find relevant information" in response["answer"]
