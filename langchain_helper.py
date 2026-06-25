from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader  
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.prompts import ChatPromptTemplate  
from langchain_community.vectorstores import DocArrayInMemorySearch  

# 1. Accept api_key as a parameter here
def create_vectordb_from_youtube(youtube_url: str, api_key: str) -> DocArrayInMemorySearch:
    # Pass the user's api_key straight to the embeddings engine
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    
    loader = YoutubeLoader.from_youtube_url(youtube_url)
    transcript = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    
    db = DocArrayInMemorySearch.from_documents(docs, embeddings) 
    return db
    
# 2. Accept api_key as a parameter here too
def get_response_from_query(db, query: str, api_key: str, k=3): 
    docs = db.similarity_search(query, k=k)
    docs_page_contents = " ".join([doc.page_content for doc in docs])
    
    # Pass the user's api_key straight to the LLM engine
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=api_key)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful YouTube assistant. 
        Based on the transcript of the video, answer the following question: {question} 
        by searching the context of the video transcript: {docs}
        
        Only use factual information. If the answer is not found in the transcript, respond with 'I don't know'. 
        Your answer should be detailed and informative."""
    )
    
    chain = prompt | llm
    response = chain.invoke({"question": query, "docs": docs_page_contents})
    
    return response.content.replace("\n", " ").strip()
