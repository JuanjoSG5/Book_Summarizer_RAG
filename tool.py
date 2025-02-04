from typing import List, Dict
from langchain_core.prompts import PromptTemplate

def create_book_summary_tool(llm, text_splitter, docs):
    """
    Create a tool to summarize the book with context preservation
    
    Args:
        llm: Language model to use for summarization
        text_splitter: Text splitter to divide the book
        docs: Original book documents
    
    Returns:
        A function that can generate summaries with context
    """
    # Split the entire book into chunks
    splits = text_splitter.split_documents(docs)
    
    def generate_section_summaries() -> List[Dict[str, str]]:
        """
        Generate summaries for each section of the book
        
        Returns:
            List of dictionaries with section summaries and their context
        """
        section_summaries = []
        
        for i in range(0, len(splits), 3):  # Process 3 chunks at a time for context
            # Select current and surrounding chunks for context
            context_chunks = splits[max(0, i-1):i+2]
            
            # Combine chunks into a single context
            full_context = "\n\n".join([chunk.page_content for chunk in context_chunks])
            
            # Create a prompt for summarization with context
            summary_prompt = PromptTemplate.from_template(
                "Given the following book context, provide a concise summary of the key points and themes:\n\n"
                "Context:\n{context}\n\n"
                "Summary:"
            )
            
            # Generate summary
            summary_chain = summary_prompt | llm
            summary = summary_chain.invoke({"context": full_context})
            
            section_summaries.append({
                "section_range": f"Chunks {i} to {i+3}",
                "context": full_context[:500] + "...",  # Truncate for brevity
                "summary": summary
            })
        
        return section_summaries
    
    def book_summary_tool(query: str = "Provide an overall summary of the book") -> str:
        """
        Main tool for book summarization
        
        Args:
            query: Optional query to guide summarization
        
        Returns:
            Comprehensive book summary
        """
        # Generate section summaries
        section_summaries = generate_section_summaries()
        
        # Create a comprehensive summary prompt
        comprehensive_summary_prompt = PromptTemplate.from_template(
            "Using the following section summaries, create a comprehensive overview of the book:\n\n"
            "{section_summaries}\n\n"
            "Additional guidance: {query}\n\n"
            "Comprehensive Summary:"
        )
        
        # Generate comprehensive summary
        comprehensive_summary_chain = comprehensive_summary_prompt | llm
        comprehensive_summary = comprehensive_summary_chain.invoke({
            "section_summaries": "\n\n".join([
                f"Section {s['section_range']}:\nContext: {s['context']}\nSummary: {s['summary']}" 
                for s in section_summaries
            ]),
            "query": query
        })
        
        return comprehensive_summary
    
    summary = book_summary_tool()
    return summary
