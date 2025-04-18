import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from azure.core.credentials import AzureKeyCredential 
from azure.search.documents import SearchClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from azure.search.documents.indexes.models import SearchIndex, SearchField, SearchFieldDataType, SimpleField, SearchableField, VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration, AzureOpenAIVectorizer,AzureOpenAIVectorizerParameters
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizableTextQuery
from summarise import *
from keywords import *
from related_works import *

def init():

    load_dotenv(override=True)

    st.session_state.llm = AzureChatOpenAI(
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_APIKEY'],
        deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
        model_name=os.environ['AZURE_OPENAI_MODEL_NAME'],
        api_version=os.environ['AZURE_OPENAI_API_VERSION'],
        temperature=0
    )
    st.session_state.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'], 
                api_key=os.environ['AZURE_OPENAI_APIKEY'], 
                model=os.environ['TEXT_EMBEDDING_MODEL_NAME'],
                azure_deployment=os.environ['TEXT_EMBEDDING_DEPLOYMENT_NAME'])
    
    st.session_state.index_search = SearchClient(
        endpoint=os.environ.get('AZURE_AI_SEARCH_ENDPOINT'), 
        index_name="iotsearchindex", 
        credential= AzureKeyCredential(os.environ.get('AZURE_AI_SEARCH_API_KEY'))
    )
 
    persona = "You are an AI research assistant specialized in analyzing and explaining academic research papers."
    task = "Your task is to provide detailed, accurate answers about research paper content including methodologies, findings, and implications, based strictly on the provided context."
    condition1 = "If the answer cannot be definitively found in the paper's content, say `I don't have sufficient information from this paper to answer that question.`"
    condition2 = "Always reference specific sections/page numbers when possible and maintain academic rigor."
    condition3 = "Explain technical terms where appropriate and highlight key contributions of the research."
    condition4 = "For comparative questions about multiple papers, clearly identify which paper you're referencing."
    ### any other things to add on

    ## Constructing initial system message
    sysmsg = f"{persona} {task} {condition1} {condition2} {condition3} {condition4}"
    st.session_state.conversation = [SystemMessage(content=sysmsg)]      
    st.session_state.messages = []
    st.session_state.init = True                               


def append_chat(role, msg):
    if role == 'human':
        st.session_state.conversation.append(HumanMessage(content=msg))
    elif role == "ai":
        st.session_state.conversation.append(AIMessage(content=msg))
    
    chat = {'role':role, 'content':msg}
    st.session_state.messages.append(chat)


def answer(query):

    # Search for relevant context in the vector store
    vector_query_content = VectorizableTextQuery(text=query, k_nearest_neighbors=50, fields="content_vector")
    context = st.session_state.index_search.search(search_text=query, vector_queries=[vector_query_content], top=4)
    contexts = ""
    for con in context:
        contexts += con['content'] + "\n" 

    context_query = f"""
        Please answer the user query.\n
        Query: {query}
        Context: {contexts}
    """

    ## Deep Copy convesation and append the context_query (retrieved contexts + latest HumanMessage) to LLM
    clone_conversation = st.session_state.conversation.copy()
    clone_conversation.append(HumanMessage(content=context_query))
    answer = st.session_state.llm.invoke(clone_conversation).content
    
    ## Store the HumanMessage & AIMessage to 
    append_chat("human", query)
    append_chat("ai",answer)

    return answer

def create_index(index_name):
    try:
        # Defines instance of SearchIndexClient class
        search_index_client = SearchIndexClient(
            os.environ.get('AZURE_AI_SEARCH_ENDPOINT'), 
            AzureKeyCredential(os.environ.get('AZURE_AI_SEARCH_API_KEY'))
        )
        
        index_fields = [
            # For ID
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
                retrievable=True,
                stored=True,
                sortable=False,
                facetable=False
            ),
            # For content word field
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=False,
                retrievable=True,
                stored=True,
                sortable=False,
                facetable=False
            ),
            # For content vector field
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,  # Use the same embedding dimension
                vector_search_profile_name="my_profile"
            )
        ]

        # Configure the vector search configuration  
        vector_search_config = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="my_algo"
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="my_profile",
                    algorithm_configuration_name="my_algo",
                    vectorizer_name="my_vectorizer"
                )
            ],
            vectorizers=[
                AzureOpenAIVectorizer(
                    vectorizer_name="my_vectorizer",
                    parameters=AzureOpenAIVectorizerParameters(
                        resource_url=os.environ.get('AZURE_OPENAI_ENDPOINT'),
                        api_key=os.environ.get('AZURE_OPENAI_APIKEY'),
                        model_name=os.environ.get('TEXT_EMBEDDING_MODEL_NAME'),
                        deployment_name=os.environ.get('TEXT_EMBEDDING_DEPLOYMENT_NAME')
                    )
                )
            ]
        )

        searchindex = SearchIndex(name=index_name, fields=index_fields, vector_search=vector_search_config)
        search_index_client.create_or_update_index(index=searchindex)
        return index_name
    except Exception as e:
        print(f"Error creating index: {e}")
        return None
    
def process_and_index_document(uploaded_file):
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
    try:
        # Load document based on file type
        if uploaded_file.name.lower().endswith('.pdf'):
            loader = PyPDFLoader(temp_file_path)
        elif uploaded_file.name.lower().endswith(('.docx', '.doc')):
            loader = Docx2txtLoader(temp_file_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Load and split documents
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "(?<=\. )", " ", ""],
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(documents)

        # Initialize progress bar
        progress_bar = st.progress(0)
        total_chunks = len(chunks)
        
        # Process chunks for Azure AI Search
        search_documents = []
        for i, chunk in enumerate(chunks):
            # Update progress bar
            progress = (i + 1) / total_chunks
            progress_bar.progress(min(progress, 1.0))
            # Generate embeddings (using OpenAI as example)
            embeddings = st.session_state.embeddings.embed_query(chunk.page_content)  # Implement your embedding function
            
            search_documents.append({
                "id": f"doc-chunk-{i}",
                "content": chunk.page_content,
                "content_vector": embeddings,
            })
        # Final update to ensure progress reaches 100%
        progress_bar.progress(1.0)
        
        st.session_state.index_search.upload_documents(documents=search_documents)
        progress_bar.empty()
        return True, f"Successfully processed! You can use the chatbot now."
    
    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(temp_file_path)  # Clean up temp file


def chatbot_page():
    
    st.title("Research paper chatbot")
    st.write("Ask questions about your uploaded research paper here in a conversational manner.")

    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    if prompt := st.chat_input("Enter your question here"):
        st.chat_message('human').markdown(prompt)
        response = answer(prompt)
        st.chat_message('ai').markdown(response)

    #st.session_state.conversation

def upload_research_paper_page():

    st.info(
        "📌 **Research Paper Upload Guidelines:**\n"
        "- Accepted formats: **PDF (.pdf) and Word Document (.docx)**\n"
        "- Ensure the file is **not password-protected**\n"
        "- Only **one file** is accepted\n"
    )

    if "messages" not in st.session_state:
        init()

    uploaded_file = st.file_uploader(
            "Upload research paper (PDF, docx only)", 
            type=["pdf", "docx"],
            accept_multiple_files=False,
            key="paper_uploader"
        )
    if st.button("Process File"):
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                    create_index("iotsearchindex")
                    success, message = process_and_index_document(uploaded_file)
                    
                    if success:
                        st.success(message)
                        st.session_state.show_upload = False
                    else:
                        st.error(f"Error: {message}")
                        st.session_state.show_upload = False

def summary_page():
    # Streamlit UI
    st.title("Research Paper Summarizer:")
    st.info("Use this page to let AI generate a structured executive summary, highlighting key insights, methodology, and findings in a concise format.")

    if "executive_summary" in st.session_state:
        st.write(st.session_state.executive_summary)
    
    else:
        if st.button("Generate Executive Summary"):
                try:
                    orchestrator_agent()
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

def keywords_page():
    # Streamlit UI
    st.title("Keyword Extractor")
    st.info("Use this page to let AI extract keywords from your uploaded research paper and display them in a word cloud.")

    if "keyword_counts" in st.session_state:
                keyword_counts = st.session_state.keyword_counts
                # Filter out low-frequency keywords
                min_frequency = 2
                filtered_counts = {k: v for k, v in keyword_counts.items() if v >= min_frequency}

                # Create visualization
                st.subheader("Research Keywords Cloud:")
                col1, col2 = st.columns([3, 1])

                with col1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='black',
                        colormap='viridis',
                        max_words=50
                    ).generate_from_frequencies(filtered_counts)
                    
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(fig)

                with col2:
                    st.markdown("**Top Keywords**")
                    for word, count in keyword_counts.most_common(10):
                        st.markdown(f"- {word} ({count})")
    else:
        if st.button("Extract Keywords"):
            try:
                generate_keyword_analysis()
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

def related_works_page():
    # Streamlit UI
    st.title("Related Works:")

    tab1, tab2 = st.tabs(["arxiV", "Semantic Scholar"])

    with tab1:
        st.info("Use this page to let AI search for other research papers related to the one uploaded on arxiV.\n"
        "May take up to a a minute to generate results.")
        if "related_papers_response_arxiv" in st.session_state:
            st.write(st.session_state.related_papers_response_arxiv["output"])
        else:
            if st.button("Get Related Works from arxiV"):
                try:
                    get_related_from_arxiv()
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    
    with tab2:
        st.info("Use this page to let AI search for other research papers related to the one uploaded on arxiV.\n"
        "May take up to a few minutes to generate results.")
        if "related_papers_response_semanticscholar" in st.session_state:
            st.write(st.session_state.related_papers_response_semanticscholar["output"])
        else:
            if st.button("Get Related Works from Semantic Scholar"):
                try:
                    get_related_from_semanticscholar()
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

def main():

    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "upload"  # Default mode

    with st.sidebar:

        if st.button("📂 Add research paper"):
            st.session_state.app_mode = "upload"
   
        st.sidebar.title("Useful tools")

        if st.button("🤖 Chat with AI"):
            st.session_state.app_mode = "chat"

        if st.button("🔍 Get Summary"):
            st.session_state.app_mode = "summary"
        
        if st.button("📌 Extract Key Terms"):
            st.session_state.app_mode = "keywords"

        #if st.button("📚 Show References"):
            #st.write("Listing cited works...")
    
        if st.button("🔗 Related Works"):
            st.session_state.app_mode = "related_works"
    
    if st.session_state.app_mode == "chat":
        chatbot_page()
    elif st.session_state.app_mode == "upload":
        upload_research_paper_page()
    elif st.session_state.app_mode == "summary":
        summary_page()
    elif st.session_state.app_mode == "keywords":
        keywords_page()
    elif st.session_state.app_mode == "related_works":
        related_works_page()

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()

