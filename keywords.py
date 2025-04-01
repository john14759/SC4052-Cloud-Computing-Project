from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

def extract_keywords_batch(text_chunks_batch):
    """Extract keywords from a batch of text chunks using LLM"""
    # Combine chunks with separator for batch processing
    combined_text = "\n\n===CHUNK SEPARATOR===\n\n".join(text_chunks_batch)
    
    batch_prompt = PromptTemplate(
        input_variables=["text"],
        template="""Extract 3-5 relevant keywords from EACH of the following research paper text chunks.
        For each chunk, return a comma-separated list of lowercase keywords, excluding general terms like 'paper' or 'study'.
        Format your response as:
        
        Chunk 1: keyword1, keyword2, keyword3
        Chunk 2: keyword1, keyword2, keyword3, keyword4
        Chunk 3: keyword1, keyword2, keyword3
        
        Text chunks:
        {text}
        
        Keywords:"""
    )
    
    keyword_chain = LLMChain(llm=st.session_state.llm, prompt=batch_prompt)
    keywords_result = keyword_chain.run({"text": combined_text})
    
    # Parse the results
    all_keywords = []
    try:
        for line in keywords_result.strip().split('\n'):
            if ':' in line:  # More robust check for keyword lines
                # Extract just the keywords part after the colon
                keywords_part = line.split(':', 1)[1]
                # Clean and validate keywords
                chunk_keywords = [k.strip().lower() for k in keywords_part.split(",") if k.strip() and len(k.strip()) > 2]
                all_keywords.extend(chunk_keywords)
    except Exception as e:
        st.warning(f"Error parsing LLM response: {str(e)}\nResponse: {keywords_result}")
        # Fallback: try to extract any keywords even if formatting is incorrect
        all_text = keywords_result.lower()
        potential_keywords = [word.strip() for word in all_text.replace('\n', ' ').split(',')]
        all_keywords = [k for k in potential_keywords if len(k) > 2 and not k.startswith('chunk')]
    
    return all_keywords

def get_all_chunks():
    """Retrieve all content chunks from Azure Cognitive Search index"""
    all_chunks = []
    results = st.session_state.index_search.search(
        search_text="*",
        select=["content"],
        top=1000  # Adjust based on your index size
    )
    for result in results:
        all_chunks.append(result["content"])
    return all_chunks

def generate_keyword_analysis():
    """Main function to handle keyword processing and visualization with batch processing"""
    status_container = st.empty()

    # Check if keywords are already stored in session_state
    if "keyword_counts" not in st.session_state:
        with st.spinner("🔍 Retrieving all document chunks..."):
            all_chunks = get_all_chunks()
    
        keyword_counts = Counter()
        progress_bar = st.progress(0)
        total_chunks = len(all_chunks)
        
        # Set batch size
        batch_size = 3
        
        # Process chunks in batches
        for i in range(0, total_chunks, batch_size):
            batch_end = min(i + batch_size, total_chunks)
            current_batch = all_chunks[i:batch_end]
            
            status_container.markdown(f"**Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}...**")
            
            try:
                batch_keywords = extract_keywords_batch(current_batch)
                keyword_counts.update(batch_keywords)
            except Exception as e:
                st.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            
            progress_bar.progress(batch_end / total_chunks)
        
        # Store results in session_state
        st.session_state.keyword_counts = keyword_counts
    else:
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

