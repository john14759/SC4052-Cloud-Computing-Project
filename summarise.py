import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from azure.search.documents.models import VectorizableTextQuery

def generate_section_query(section_name):
    """Generate an optimized search query for a research paper section"""
    query_prompt_template = PromptTemplate(
        input_variables=["section_name"],
        template="""Generate an optimized search query to retrieve relevant content from a vector store for the {section_name} section of a research paper. 
        Ensure that the query includes commonly used keywords commonly to describe the {section_name} title in research papers.
        
        Query: [List of keywords]

        Return the generated query only without any additional text.
        """
    )
    query_generator = LLMChain(llm=st.session_state.llm, prompt=query_prompt_template)
    generated_query = query_generator.run({"section_name": section_name})
    print(f"Generated Query for {section_name}: {generated_query}")

    return generated_query

def get_section_agent(section_name):
    """Get appropriate LLMChain agent for a given section"""
    section_prompts = {
        "abstract": PromptTemplate(
            input_variables=["text"],
            template="You are a given chunks of text from the abstract of a research paper. Generate a comprehensive summary:\n{text}"
        ),
        "introduction": PromptTemplate(
            input_variables=["text"],
            template="You are a given chunks of text from the introduction of a research paper. Generate a comprehensive summary:\n{text}"
        ),
        "conclusion": PromptTemplate(
            input_variables=["text"],
            template="You are a given chunks of text from the conclusion of a research paper. Generate a comprehensive summary:\n{text}"
        )
    }

    return LLMChain(llm=st.session_state.llm, prompt=section_prompts[section_name])

def extract_relevant_chunks(section_name):
    """Retrieve relevant content chunks using hybrid search"""
    search_query = generate_section_query(section_name)
    
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

    chunks = [result["content"] for result in results]
    return chunks

def summarize_section(section_name):
    """Retrieve and summarize content for a specific section"""
    chunks = extract_relevant_chunks(section_name)
    agent = get_section_agent(section_name)
    summary = agent.run({"text": " ".join(chunks)})
    print(f"Summary for {section_name}: {summary}")
    return summary

def generate_executive_summary(summaries):
    """Generate final executive summary from section summaries"""
    final_prompt = PromptTemplate(
        input_variables=["abstract", "introduction", "conclusion"],
        template="""
        Based on the following excerpts from a research paper, generate a comprehensive executive summary:

        ABSTRACT: {abstract}
        INTRODUCTION: {introduction}
        CONCLUSION: {conclusion}

        Please create a structured executive summary (400-600 words) that includes:
        1. The main research question or problem addressed
        2. Key methodology and approach
        3. Most significant findings and their implications
        4. Practical applications or recommendations
        5. Limitations acknowledged in the research

        Format the summary with clear headings and bullet points where appropriate. 
        The tone should be formal and objective, suitable for busy executives who need to quickly understand the core value of this research.
        """
    )
    summary_chain = LLMChain(llm=st.session_state.llm, prompt=final_prompt)
    executive_summary = summary_chain.run(summaries)
    return executive_summary

def orchestrator_agent():
    """Main coordination function for document processing"""
    header_container = st.empty()  # Placeholder for subheader
    progress_container = st.empty()  # Placeholder for progress bar
    status_text = st.empty()  # Placeholder for status updates

    # Show subheader and progress bar initially
    header_container.subheader("🚀 Research Paper Processing")
    progress_bar = progress_container.progress(0)

    summaries = {}
    sections = ["abstract", "introduction", "conclusion"]

    for i, section in enumerate(sections):
        status_text.write(f"**Processing {section}...**")
        summaries[section] = summarize_section(section)
        progress_bar.progress((i + 1) / 4)

    # Generate final summary
    status_text.write("**Generating Executive Summary...**")
    if "executive_summary" not in st.session_state:
        st.session_state.executive_summary = generate_executive_summary(summaries)
    progress_bar.progress(1.0)

    # Remove subheader and progress bar
    header_container.empty()
    progress_container.empty()
    status_text.empty()
    st.rerun()


