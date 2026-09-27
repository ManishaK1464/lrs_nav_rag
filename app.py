"""Step 3: simple chat UI.

    streamlit run app.py
"""
import streamlit as st

from rag import RAG

st.set_page_config(page_title="LRS-Nav docs assistant")
st.title("LRS-Nav docs assistant")
st.caption("Ask about the project. Answers come only from the indexed documents and run fully on CPU.")


@st.cache_resource(show_spinner="Loading models (first time takes a minute)...")
def get_rag():
    return RAG()


rag = get_rag()

if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    st.chat_message(role).write(text)

question = st.chat_input("e.g. What happens when the Reviewer rejects the code?")
if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and writing the answer..."):
            answer, docs = rag.answer(question)
        st.write(answer)
        with st.expander("Sources used"):
            for i, d in enumerate(docs, 1):
                st.markdown(f"**[{i}] {d.metadata.get('source', '?')}**")
                st.text(d.page_content[:500])
    st.session_state.history += [("user", question), ("assistant", answer)]
