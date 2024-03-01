import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage  # Components for chat interaction
from langchain_community.document_loaders import WebBaseLoader  # To fetch webpage contents
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To segment text into smaller parts
from langchain_community.vectorstores import Chroma  # For storing text vectors
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # Utilizing OpenAI for models and embeddings
from dotenv import load_dotenv  # To manage environment variables
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Templates for generating chat prompts
from langchain.chains import create_history_aware_retriever, create_retrieval_chain  # Logic for information retrieval
from langchain.chains.combine_documents import create_stuff_documents_chain  # Logic for merging document information

load_dotenv()  # Initialize environment variables

# Specify your OpenAI API key
api_key = ''

def retrieve_and_vectorize_web_content(url): 
    # Process webpage content
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # Divide the document into manageable segments
    splitter = RecursiveCharacterTextSplitter()
    document_parts = splitter.split_documents(document)
    
    # Create vector storage for document segments
    vector_store = Chroma.from_documents(document_parts, OpenAIEmbeddings(api_key=api_key))

    return vector_store

def construct_contextual_retrieval_chain(vector_store):
    model = ChatOpenAI(api_key=api_key)  # Initialize the language model
    
    # Establish a retriever for fetching relevant text segments
    retriever = vector_store.as_retriever()
    
    # Define chat interaction prompts
    chat_prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Considering our discussion, what should we look into?")
    ])
    
    # Create a retrieval chain that considers chat history
    history_informed_chain = create_history_aware_retriever(model, retriever, chat_prompt)
    
    return history_informed_chain
    
# Function to incorporate chat logic into the retrieval process
def embed_chat_logic_into_retrieval(history_informed_chain): 
    model = ChatOpenAI(api_key=api_key)
    
    # Configure prompts for utilizing provided context in conversation
    context_prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the query with this context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    document_combiner_chain = create_stuff_documents_chain(model, context_prompt)
    
    return create_retrieval_chain(history_informed_chain, document_combiner_chain)

def obtain_chat_reply(user_input):
    retrieval_chain = construct_contextual_retrieval_chain(st.session_state.vector_store)
    combined_chain = embed_chat_logic_into_retrieval(retrieval_chain)
    response = combined_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

# Initialize Streamlit app settings
st.set_page_config(page_title="Web Content AI Scraper!", page_icon="🌍")
st.title("Web Scraper AI 🌍")

# Side panel for configurations
with st.sidebar:
    st.header("Settings")
    web_url_input = st.text_input("Website URL")

if not web_url_input:
    st.info("Enter a website URL to begin.")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello! How may I help you today?")]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = retrieve_and_vectorize_web_content(web_url_input)    

    user_query = st.chat_input("What information do you need?")
    if user_query:
        reply = obtain_chat_reply(user_query)
        st.session_state.chat_history.extend([HumanMessage(content=user_query), AIMessage(content=reply)])

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        else:
            with st.chat_message("User"):
                st.write(message.content)


# cd to github/rag-project run the command below in your terminal to start the program
# streamlit run app.py