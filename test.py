from os import getenv
from dotenv import load_dotenv
import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_core.tools import tool
from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


load_dotenv()

# TODO: El RAG es para preguntas especificas del libro, mientrass que la tool es para resumir el libro
# TODO: Para el resumen del libro, divide el texto en fragmentos y crea resumenes de cada fragmento, 
# esto da un problema que es que se pierde el contexto del libro. Por lo que tengo que pasarle contexto sobre 
# lo que va de las partes en las que se produce esta separacion 

# 1. Inicializamos el modelo de ChatOpenAI
llm = ChatOpenAI(
    openai_api_key=getenv("OPENROUTER_API_KEY"),
    openai_api_base=getenv("OPENROUTER_BASE_URL"),
    model_name="openai/gpt-4o",
    model_kwargs={
        "extra_headers": {
            "Helicone-Auth": f"Bearer " + getenv("HELICONE_API_KEY")
        }
    },
)

# 2. Cargamos el libro que queremos usar de ejemplo
url = "sample.txt"

loader = TextLoader( url)
docs = loader.load()

# 3. Dividimos el texto en fragmentos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)


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
    
    return book_summary_tool



    
    

# 4. Inicializamos Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 5. Creamos un vector store con Chroma
vectorstore = Chroma(embedding_function=embeddings)
vectorstore.add_documents(splits)

# 6. Función principal del chatbot (responde en streaming y muestra en consola los trozos usados)
def chatbot(message, history):
    
    # Dentro del almacen de vecotres buscamos las partes más parecidas
    relevant_docs = vectorstore.similarity_search(message)

    print("\n=== Fragmentos de documento utilizados para la respuesta ===")
    for i, doc in enumerate(relevant_docs, 1):
        texto = doc.page_content.replace("\n", " ")
        print(f"\nFragmento {i}:\n{texto[:300]}...")

    # Unimos el contenido de la pagina de los documentos relevantes para crear un contexto
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Creamos el prompt final, que no se muestra en la interfaz
    final_prompt = (
        "Eres un asistente experto en solucionar dudas sobre conceptos descritos en un libro. "
        "Utiliza el siguiente contexto para responder de forma breve y concisa. "
        "Si no encuentras la información, responde que no la conoces.\n\n"
        f"Contexto:\n{context_text}\n\n"
        f"Pregunta: {message}\n"
        "Respuesta:"
    )

    messages = []
    # hacemos la funcion de añadir mensajes al historial, para que no se pierdan si reiniciamos el chat
    for chat_message in history:
        messages.append(chat_message)
    messages.append({"role": "user", "content": final_prompt})

    # Con la llamada a la API, obtenemos la respuesta en streaming
    response = llm.stream(messages)
    # Inicializa una variable para almacenar la respuesta parcial
    partial_response = ""

    # Itera sobre cada fragmento en la respuesta
    for chunk in response:
        # Verifica si el fragmento no está vacío y tiene el atributo "content"
        if chunk and hasattr(chunk, "content"):
            content = chunk.content
            # Si el contenido no es None, lo añade a la respuesta parcial
            if content is not None:
                partial_response += content
                # Genera la respuesta parcial actualizada
                yield partial_response

# 7. Interfaz de Gradio
# Modify your Gradio interface to include a summary button
def create_interface():
    with gr.Blocks as demo:
        gr.ChatInterface(
            chatbot,
            chatbot=gr.Chatbot(height=400, type="messages"),
            textbox=gr.Textbox(placeholder="Escribe tu mensaje aquí...", container=False, scale=7),
            title="ChatBot RAG - Resumen de libros",
            description="Asistente virtual resumidor de libros.",
            theme="ocean",
            examples=[
                "¿De que trata el libro?",
                "¿Cuál es el argumento principal del libro?",
                "¿Qué parte puedes destacar del libro?"
            ],
            type="messages", 
            editable=True,
            save_history=True,
        )
        gr.Checkbox(label="Morning", info="Did they do it in the morning?")
    return demo
        

# Replace your previous launch code with this
if __name__ == "__main__":
    demo = create_interface()
    demo.queue().launch()
