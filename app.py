import streamlit as st

def main():
    st.set_page_config(
        page_title="DocGPT",
        page_icon="📚",
        layout="wide"
    )
    st.title("Welcome to DocGPT 📚")
    st.header("Your AI-powered document assistant")
    st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Your Documents")
        st.file_uploader("Upload your PDFs here and click on 'Process Documents'")
        st.button("Process Documents")


if __name__ == "__main__":
    main()