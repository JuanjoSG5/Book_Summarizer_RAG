import os
from os import getenv
from dotenv import load_dotenv
import openai
import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tool import create_book_summary_tool
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

llm = ChatOpenAI(
    openai_api_key=getenv("OPENROUTER_API_KEY"),
    openai_api_base=getenv("OPENROUTER_BASE_URL"),
    model_name="google/gemini-flash-1.5",
    model_kwargs={
        "extra_headers": {
            "Helicone-Auth": f"Bearer {getenv('HELICONE_API_KEY')}"
        }
    },
)

# Initialize common resources for file processing
textSplitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=1000)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def process_uploaded_file(file):
    """
    Process the uploaded file by loading its content, splitting the text,
    and creating a vectorstore for retrieval. Supports both text and PDF files.
    """
    if file is None:
        return None, None
    # Obtain file path from file object or path string
    file_path = file.name if hasattr(file, 'name') else file
    _, ext = os.path.splitext(file_path)
    
    # Select the appropriate loader based on file extension
    if ext.lower() == '.pdf':
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    docs = loader.load()
    splits = textSplitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return docs, vectorstore

def process_file(file):
    """
    Single function to process the uploaded file, returning docs, vectorstore, and a status message.
    """
    docs, vectorstore = process_uploaded_file(file)
    if docs is None or vectorstore is None:
        return None, None, "Error al procesar el archivo. Asegúrate de que el archivo es válido."
    return docs, vectorstore, "Archivo procesado exitosamente."

def generateBookSummary(docs):
    """
    Generate book summary using the tool with the provided documents.
    """
    return create_book_summary_tool.invoke({
        "llm": llm, 
        "text_splitter": textSplitter, 
        "docs": docs
    })

def chatbot(message, previousHistory, vectorstore):
    """
    Build the prompt with conversation history and file-based context, then stream the LLM response.
    """
    relevantDocs = vectorstore.similarity_search(message)
    contextText = "\n\n".join([doc.page_content for doc in relevantDocs])
    
    finalPrompt = (
        "Eres un asistente experto en solucionar dudas sobre conceptos descritos en un libro. "
        "Utiliza el siguiente contexto para responder de forma breve y concisa. "
        "Si no encuentras la información, responde que no la conoces.\n\n"
        f"Contexto:\n{contextText}\n\n"
        f"Pregunta: {message}\n"
        "Respuesta:"
    )

    messages = [{"role": "user", "content": finalPrompt}]
    
    response = llm.stream(messages)
    partialResponse = ""
    for chunk in response:
        if chunk and hasattr(chunk, "content"):
            content = chunk.content
            if content is not None:
                partialResponse += content
                yield partialResponse

def processMessage(message, history, vectorstore, docs):
    """
    Process incoming messages. If the message contains "resumen", generate a book summary;
    otherwise, stream the chatbot response using the uploaded file’s vectorstore.
    """
    new_history = history + [{"role": "user", "content": message}]
    displayHistory = new_history + [{"role": "assistant", "content": "⏳ Procesando..."}]
    yield displayHistory, new_history, vectorstore, docs
    
    keywords = ["resumen", "resume"]

    # Check if file processing has been done
    if vectorstore is None or docs is None:
        error_msg = "Por favor, sube un archivo válido antes de enviar mensajes."
        updatedHistory = new_history + [{"role": "assistant", "content": error_msg}]
        yield updatedHistory, new_history, vectorstore, docs
        return

    if any(keyword in message.lower() for keyword in keywords):
        summary = generateBookSummary(docs)
        updatedHistory = new_history + [{"role": "assistant", "content": summary.content}]
        yield updatedHistory, new_history, vectorstore, docs
    else:
        full_response = ""
        for response_chunk in chatbot(message, new_history, vectorstore):
            full_response = response_chunk
            updatedDisplayHistory = new_history + [{"role": "assistant", "content": full_response}]
            yield updatedDisplayHistory, new_history, vectorstore, docs

def createSummary(history, docs):
    summary = generateBookSummary(docs)
    return history + [{"role": "assistant", "content": summary.content}]

def createInterface():
    with gr.Blocks(title="ChatBot RAG - Resumen de libros", theme="ocean") as demo:
        gr.Markdown("## Asistente virtual resumidor de libros")
        
        # States to store chat history, document data, and the vectorstore
        browser_state = gr.BrowserState(default_value=[], storage_key="chat_history")
        docs_state = gr.State(None)
        vector_state = gr.State(None)
        
        # File upload component for user-provided book file
        file_upload = gr.File(label="Sube tu archivo de libro", file_count="single")
        file_process_button = gr.Button("Procesar Archivo")
        file_status = gr.Textbox(label="Estado de archivo", interactive=False)
        
        # Chatbot interface components
        chatbotComponent = gr.Chatbot(type="messages", height=600)
        textbox = gr.Textbox(placeholder="Escribe tu mensaje aquí...", container=False, scale=7)
        
        with gr.Row():
            submitBtn = gr.Button("Enviar")
            
        examples = gr.Examples(
            examples=[
                "¿Cuál es el argumento principal?",
                "Haz un resumen del contenido",
                "¿Qué temas principales se abordan?"
            ],
            inputs=[textbox],
            label="Ejemplos:"
        )

        # Process the uploaded file and update states in one function call.
        file_process_button.click(
            fn=process_file,
            inputs=[file_upload],
            outputs=[docs_state, vector_state, file_status]
        )
        
        submitEvent = textbox.submit(
            fn=processMessage,
            inputs=[textbox, browser_state, vector_state, docs_state],
            outputs=[chatbotComponent, browser_state, vector_state, docs_state],
            show_progress="hidden"
        )
        submitEvent.then(lambda: gr.Textbox(value=""), None, [textbox])
        
        submitBtn.click(
            fn=processMessage,
            inputs=[textbox, browser_state, vector_state, docs_state],
            outputs=[chatbotComponent, browser_state, vector_state, docs_state],
            show_progress="hidden"
        ).then(lambda: gr.Textbox(value=""), None, [textbox])

    return demo

if __name__ == "__main__":
    demo = createInterface()
    demo.launch()
