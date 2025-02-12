from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

@tool
def createBookSummaryTool(llm, text_splitter, docs):
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
    
    def generateSectionSummaries() -> List[Dict[str, str]]:
        """
        Generate summaries for each section of the book
        
        Returns:
            List of dictionaries with section summaries and their context
        """
        sectionSummaries = []
        
        # Iterate over 2 chunks at the time, to reduce the context size
        for i in range(0, len(splits), 2):  
            # Select current and surrounding chunks for context
            contextChunks = splits[max(0, i-1):i+2]
            
            # Combine chunks into a single context
            fullContext = "\n\n".join([chunk.page_content for chunk in contextChunks])

            prompt = PromptTemplate.from_template(
                "Given the following book context, provide a concise summary of the key points and themes:\n\n"
                "Context:\n{context}\n\n"
                "Summary:"
            )
            
            # Generate summary
            summaryChain = prompt | llm
            summary = summaryChain.invoke({"context": fullContext})
            
            sectionSummaries.append({
                "section_range": f"Chunks {i} to {i+2}",
                "context": fullContext[:500] + "...", 
                "summary": summary
            })
        
        return sectionSummaries
    
    def bookSummaryTool(query: str = "Provide an overall summary of the book") -> str:
        """
        Main tool for book summarization
        
        Args:
            query: Optional query to guide summarization
        
        Returns:
            Comprehensive book summary
        """
        # Generate section summaries
        sectionSummaries = generateSectionSummaries()

        prompt = PromptTemplate.from_template(
            "Using the following section summaries, create a comprehensive overview of the book:\n\n"
            "{section_summaries}\n\n"
            "Additional guidance: {query}\n\n"
            "Comprehensive Summary:"
        )
        
        # Generate a summary for all of the sections
        totalSummaryChain = prompt | llm
        summary = totalSummaryChain.invoke({
            "section_summaries": "\n\n".join([
                f"Section {s['section_range']}:\nContext: {s['context']}\nSummary: {s['summary']}" 
                for s in sectionSummaries
            ]),
            "query": query
        })
        
        return summary
    
    summary = bookSummaryTool()
    return summary
