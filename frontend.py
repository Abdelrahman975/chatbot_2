from model import run
import streamlit as st
from streamlit_chat import message as st_message

# Set page config
st.set_page_config(
    page_title="مساعد الأسئلة الشائعة",
    page_icon="❓",
    layout="centered"
)

# Add custom CSS for RTL support
st.markdown("""
<style>
    body {
        direction: rtl;
    }
    .stTextInput, .stButton {
        text-align: right;
    }
    .css-1kyxreq {
        justify-content: flex-end;
    }
</style>
""", unsafe_allow_html=True)

# Page header
st.title("مساعد الأسئلة الشائعة")
st.markdown("أهلاً بك! يمكنك طرح أي سؤال وسأحاول الإجابة عليه من قاعدة بيانات الأسئلة الشائعة.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("اكتب سؤالك هنا..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show a spinner while processing
    with st.spinner("جاري البحث عن إجابة..."):
        response = run(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})