from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_community.utilities import ArxivAPIWrapper
from langchain.tools import Tool
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from azure.search.documents.models import VectorizableTextQuery

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

    return result["text"].strip()

def get_related_from_arxiv():

    if "related_papers_response" not in st.session_state:
        title_keyword = extract_keywords()
        print(title_keyword)
        st.session_state.related_papers_response = None

        # Create the ArxivAPIWrapper instance
        arxiv_wrapper = ArxivAPIWrapper(
            top_k_results=5,
            ARXIV_MAX_QUERY_LENGTH=300,
            load_max_docs=5,
            load_all_available_meta=False,
            doc_content_chars_max=40000
        )

        # Create a Tool with the ArxivAPIWrapper
        arxiv_tool = Tool(
            name="arxiv",
            func=arxiv_wrapper.run,
            description="Useful for when you need to answer questions about scientific research papers. The input should be a search query."
        )

        tools = [arxiv_tool]
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(st.session_state.llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        st.session_state.related_papers_response = agent_executor.invoke({
            "input": f"Give me a list of research papers about {title_keyword}. Format the summary with clear headings and bullet points where appropriate. ",
        })


