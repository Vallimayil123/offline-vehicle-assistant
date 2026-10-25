
from pathlib import Path
import platform
import sys

import streamlit as st

from src.assistant import VehicleAssistant
from src.ingest import ManualIngestor


PROJECT_ROOT = Path(__file__).resolve().parent


def get_project_status():
    return {
        "project": "Offline AI Vehicle Assistant",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "project_folder_exists": PROJECT_ROOT.is_dir(),
        "manuals_folder_exists": (
            PROJECT_ROOT / "data" / "manuals"
        ).is_dir(),
    }


@st.cache_resource
def get_ingestor():
    return ManualIngestor()


@st.cache_resource
def get_assistant():
    return VehicleAssistant(ingestor=get_ingestor())


def main():
    st.set_page_config(
        page_title="Offline AI Vehicle Assistant",
        page_icon="🚗",
        layout="wide",
    )

    st.title("🚗 Offline AI Vehicle Assistant")
    st.caption(
        "Ask questions about your car manuals using a local AI model."
    )

    with st.sidebar:
        st.header("Vehicle Manuals")
        st.write("Upload a text-based PDF owner manual to index it.")

        uploaded_file = st.file_uploader(
            "Choose a PDF manual",
            type=["pdf"],
        )

        if uploaded_file is not None:
            if st.button("Process manual", type="primary"):
                manuals_dir = PROJECT_ROOT / "data" / "manuals"
                manuals_dir.mkdir(parents=True, exist_ok=True)

                safe_name = Path(uploaded_file.name).name
                pdf_path = manuals_dir / safe_name

                try:
                    pdf_path.write_bytes(uploaded_file.getvalue())

                    with st.spinner("Processing manual..."):
                        summary = get_ingestor().ingest_pdf(pdf_path)

                    st.success("Manual processed successfully!")
                    st.write(f"Pages with text: {summary['pages_with_text']}")
                    st.write(f"Chunks created: {summary['chunks_created']}")
                    st.write(f"Total chunks in database: {summary['chunks_stored']}")

                except Exception as exc:
                    st.error(f"Could not process this PDF: {exc}")

        st.divider()
        st.caption("AI model: qwen2.5:3b")
        st.caption("Embeddings: all-MiniLM-L6-v2")
        st.caption("Documents are stored locally.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("View manual sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**{source['source']} — page {source['page']}**"
                        )
                        st.write(source["text"])

    question = st.chat_input("Ask a question about your vehicle...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching manuals and generating an answer..."):
                    response = get_assistant().answer(question)

                st.markdown(response["answer"])

                if response["sources"]:
                    with st.expander("View manual sources"):
                        for source in response["sources"]:
                            st.markdown(
                                f"**{source['source']} — page {source['page']}**"
                            )
                            st.write(source["text"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"],
                    }
                )

            except Exception as exc:
                st.error(
                    "The assistant could not complete the request. "
                    f"Check that Ollama is running. Details: {exc}"
                )


if __name__ == "__main__":
    main()
