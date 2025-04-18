from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, create_openai_functions_agent
from langchain_community.utilities import ArxivAPIWrapper
from langchain.tools import Tool
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from azure.search.documents.models import VectorizableTextQuery
import time

def generate_title_query():
    """Generate search query for finding paper title"""
    query_prompt_template = PromptTemplate(
        input_variables=[],
        template="""Generate an optimized search query to find the title of a research paper. 
        Focus on terms that typically appear in title sections or document metadata.
        
        Query: [List of keywords]

        Return the generated query only without any additional text.
        """
    )
    query_generator = LLMChain(llm=st.session_state.llm, prompt=query_prompt_template)
    return query_generator.run({})

def extract_title():
    """Retrieve and extract paper title from search results"""
    search_query = generate_title_query()
    
    vector_query = VectorizableTextQuery(
        text=search_query,
        k_nearest_neighbors=50,
        fields="content_vector"
    )

    results = st.session_state.index_search.search(
        search_text=search_query,
        vector_queries=[vector_query],
        select=["content"],
        top=3
    )

    # Combine results with metadata priority
    chunks = [result["content"] for result in results]
    
    # Use LLM to identify the actual title
    extract_prompt = PromptTemplate(
        input_variables=["chunks"],
        template="""Identify the main research paper title from the index search results:
        {chunks}
        Return ONLY the title without quotation marks or additional text."""
    )
    extract_chain = LLMChain(llm=st.session_state.llm, prompt=extract_prompt)
    result = extract_chain.invoke({"chunks": "\n".join(chunks)})
    print("Title:", result["text"])

    return result["text"].strip()

def extract_keywords():

    title = extract_title()

    """Extract search keywords from paper title"""
    keyword_prompt = PromptTemplate(
        input_variables=["title"],
        template="""Extract a keyword phrase from the title provided. 
        Focus on terms useful for finding related research papers.
        
        Title: {title}
        Keywords:"""
    )
    keyword_chain = LLMChain(llm=st.session_state.llm, prompt=keyword_prompt)
    result = keyword_chain.invoke({"title": title})
    print("Keywords:", result["text"])

    return result["text"].strip()

def get_related_from_arxiv():
    if "related_papers_response_arxiv" not in st.session_state:
        # Create a progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Step 1: Extract title keywords
        progress_text.text("Extracting keywords from paper title...")
        progress_bar.progress(10)
        
        title_keyword = extract_keywords()
        progress_bar.progress(30)
        progress_text.text("Keywords extracted. Setting up arXiv search...")
        
        st.session_state.related_papers_response_arxiv = None
        
        # Step 2: Set up arXiv tools
        arxiv_wrapper = ArxivAPIWrapper(
            top_k_results=5,
            ARXIV_MAX_QUERY_LENGTH=300,
            load_max_docs=5,
            load_all_available_meta=False,
            doc_content_chars_max=40000
        )
        
        arxiv_tool = Tool(
            name="arxiv",
            func=arxiv_wrapper.run,
            description="Useful for when you need to answer questions about scientific research papers. The input should be a search query."
        )
        
        tools = [arxiv_tool]
        progress_bar.progress(50)
        progress_text.text("Setting up search agent...")
        
        # Step 3: Set up the agent
        instructions = """You are an expert researcher. Your task is to search for related research papers on arXiv"""
        base_prompt = hub.pull("langchain-ai/openai-functions-template")
        prompt = base_prompt.partial(instructions=instructions)
        agent = create_openai_functions_agent(st.session_state.llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        progress_bar.progress(70)
        progress_text.text("Searching for related papers on arXiv...")
        
        # Step 4: Execute search
        st.session_state.related_papers_response_arxiv = agent_executor.invoke({
            "input": f"Give me a list of research papers about {title_keyword}."
            "Format the list with clear headings and bullet points where appropriate and only return the list."
            "Break down the task into subtasks for search if needed. Use the search tool"
        })
        
        # Complete the progress bar
        progress_bar.progress(100)
        progress_text.text("Search completed!")
        
        # Optional: Clear the progress indicators after a delay
        time.sleep(1)
        progress_text.empty()
        st.rerun()

def get_related_from_semanticscholar():

    if "related_papers_response_semanticscholar" not in st.session_state:

        # Create a progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Step 1: Extract title keywords
        progress_text.text("Extracting keywords from paper title...")
        progress_bar.progress(10)
        
        title_keyword = extract_keywords()
        progress_bar.progress(30)
        progress_text.text("Keywords extracted. Setting up SemanticScholar search...")

        st.session_state.related_papers_response_semanticscholar = None

        instructions = """You are an expert researcher. Your task is to search for related research papers on SemanticScholar"""
        base_prompt = hub.pull("langchain-ai/openai-functions-template")
        prompt = base_prompt.partial(instructions=instructions)

        tools = [SemanticScholarQueryRun()]
        progress_bar.progress(50)
        progress_text.text("Setting up search agent...")

        agent = create_openai_functions_agent(st.session_state.llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )

        progress_bar.progress(70)
        progress_text.text("Searching for related papers on SemanticScholar...")

        st.session_state.related_papers_response_semanticscholar = agent_executor.invoke(
            {
                "input": f"Give me a list of research papers about {title_keyword}."
                "Format the list with clear headings and bullet points where appropriate and only return the list."
                "Break down the task into subtasks for search if needed. Use the search tool"
            }
        )

        # Complete the progress bar
        progress_bar.progress(100)
        progress_text.text("Search completed!")
        
        # Optional: Clear the progress indicators after a delay
        time.sleep(1)
        progress_text.empty()
        st.rerun()