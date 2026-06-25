import streamlit as st
import langchain_helper as lch

st.title("YouTube Assistant")

# 1. Create a secure input field in the sidebar for the user's API key
with st.sidebar:
    user_api_key = st.text_input(
        "Enter your Google Gemini API Key",
        type="password",
        placeholder="AIzaSy...",
        help="You can get a free API key from Google AI Studio."
    )

youtube_url = st.text_input("Enter YouTube Video URL")
query = st.text_input("Ask a question about the video")

if st.button("Submit"):
    # 2. Check if the user forgot to enter their key
    if not user_api_key:
        st.error("Please provide your Google Gemini API Key in the sidebar to proceed.")
    elif not youtube_url or not query:
        st.warning("Please provide both a YouTube URL and a question.")
    else:
        with st.spinner("Processing video transcript..."):
            # 3. Pass the user's API key into the backend functions
            db = lch.create_vectordb_from_youtube(youtube_url, user_api_key)
            response = lch.get_response_from_query(db, query, user_api_key)
            
            st.subheader("Answer:")
            st.write(response)
