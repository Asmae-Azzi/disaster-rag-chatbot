import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import boto3
from io import BytesIO
from pdfminer.high_level import extract_text_from_fp # Import for PDF parsing

# --- 1. Configuration:

required_secrets = ["openai_api_key", "aws_access_key_id", "aws_secret_access_key", "aws_region_name", "s3_bucket_name"]
for secret in required_secrets:
    if secret not in st.secrets:
        st.error(f"Missing secret: '{secret}'. Please add it to Streamlit secrets.")
        st.stop()

# Set environment variables for LangChain/OpenAI
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

# AWS S3 Configuration
AWS_ACCESS_KEY_ID = st.secrets["aws_access_key_id"]
AWS_SECRET_ACCESS_KEY = st.secrets["aws_secret_access_key"]
AWS_REGION_NAME = st.secrets["aws_region_name"]
S3_BUCKET_NAME = st.secrets["s3_bucket_name"]

# --- Function to load documents from S3 ---
@st.cache_data # Caching the S3 document loading to avoid re-downloading on every rerun
def load_documents_from_s3(bucket_name, aws_access_key_id, aws_secret_access_key, aws_region_name):
    """
    Loads PDF documents from a specified S3 bucket, extracts text, and returns it.
    """
    st.write(f"Connecting to S3 bucket: {bucket_name}...")
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region_name
    )
    
    documents_raw_content = []
    
    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            st.write(f"Found {len(response['Contents'])} objects in S3. Processing PDFs...")
            for obj in response['Contents']:
                key = obj['Key']
                # Only process .pdf files
                if key.lower().endswith('.pdf'):
                    st.write(f"Downloading and parsing: {key}")
                    try:
                        obj_data = s3.get_object(Bucket=bucket_name, Key=key)
                        # Use BytesIO to create a file-like object from the S3 response body
                        with BytesIO(obj_data['Body'].read()) as pdf_file:
                            text_content = extract_text_from_fp(pdf_file)
                            documents_raw_content.append(text_content)
                            st.write(f"Successfully parsed: {key}")
                    except Exception as e:
                        st.error(f"Error parsing PDF '{key}': {e}")
                else:
                    st.info(f"Skipping non-PDF file: {key}")
        else:
            st.warning(f"No objects found in S3 bucket: {bucket_name}")
    except Exception as e:
        st.error(f"Error listing objects in S3 bucket: {e}")
        st.stop()
    
    return documents_raw_content

# --- 2. RAG System Setup ---

@st.cache_resource # Caching the RAG system to avoid rebuilding on every rerun
def setup_rag_system(documents_raw_content):
    """
    Sets up the RAG system: chunks documents, creates embeddings,
    and builds a FAISS vector store.
    """
    if not documents_raw_content:
        st.error("No documents loaded to build the RAG system. Please check your S3 bucket and PDF files.")
        st.stop()

    st.write("Initializing RAG system components...")
    
    # 1. Initialize Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    # 2. Create Documents from S3 PDF content
    from langchain.docstore.document import Document
    docs = []
    for i, content in enumerate(documents_raw_content):
        # Creating a Document object for each PDF content
        docs.append(Document(page_content=content, metadata={"source": f"S3 PDF Document {i+1}"}))

    # 3. Split documents into chunks
    chunks = text_splitter.split_documents(docs)
    st.write(f"Split {len(documents_raw_content)} S3 PDF documents into {len(chunks)} chunks.")

    # 4. Initialize OpenAI Embeddings (cheapest efficient model)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 5. Build FAISS Vector Store from chunks and embeddings
    
    st.write("Building FAISS vector store. This might take a while for many documents...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    st.write("FAISS vector store built.")

    # 6. Initialize ChatOpenAI LLM (cost-effective model)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

    # 7. Create RetrievalQA chain
    qa_template = """
    You are a helpful chatbot specializing in disaster preparedness and resilience.
    Use the following retrieved context to answer the user's question.
    If you don't know the answer based on the provided context, state that you don't know,
    and suggest they consult official emergency services or local authorities.
    Keep your answers concise and directly relevant to the question.

    Context: {context}

    Question: {question}

    Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(qa_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )
    st.success("RAG system ready!")
    return qa_chain

# --- 3. Streamlit UI ---

st.set_page_config(page_title="Disaster Preparedness Chatbot", layout="centered")

st.title("💬 Disaster Preparedness Chatbot")
st.markdown(
    """
    Ask me anything about disaster preparedness and resilience based on our curated documents.
    """
)

# Initialize chat history in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load documents from S3 and set up RAG chain
if "qa_chain" not in st.session_state:
    with st.spinner("Loading documents from S3 and setting up RAG system..."):
        documents_raw_content = load_documents_from_s3(
            S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME
        )
        st.session_state.qa_chain = setup_rag_system(documents_raw_content)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "source" in message and message["source"]:
            st.caption(f"Source: {message['source']}")

# React to user input
if prompt := st.chat_input("How can I prepare for a hurricane?"):
    # Display user message in chat history
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        # Get response from RAG chain
        response = st.session_state.qa_chain.invoke({"query": prompt})
        answer = response["result"]
        source_documents = response.get("source_documents", [])
        
        # Extract source information
        sources_info = []
        if source_documents:
            for doc in source_documents:
                # Use the 'source' metadata we set during document loading
                if doc.metadata and "source" in doc.metadata:
                    sources_info.append(doc.metadata["source"])
            if sources_info:
                sources_str = ", ".join(sorted(list(set(sources_info)))) # Use set to avoid duplicates
            else:
                sources_str = "No specific source found"
        else:
            sources_str = "No specific source found"

    # Display assistant response in chat history
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(f"Source Documents: {sources_str}")
    st.session_state.messages.append({"role": "assistant", "content": answer, "source": sources_str})

# Add a clear chat button
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun

st.markdown("---")
st.info(
    """
    **Disclaimer:** This chatbot provides general information for disaster preparedness
    and resilience based on a limited set of documents. Always consult official
    emergency services, local authorities, and expert advice for specific,
    up-to-date, and critical information relevant to your location and situation.
    """
)
