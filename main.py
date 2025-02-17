from os import getenv
from dotenv import load_dotenv
from tool import createBookSummaryTool
import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredFileLoader

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

textSplitter = RecursiveCharacterTextSplitter(chunk_size=4500, chunk_overlap=750)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def processUploadedFile(file):
    """
    Process the uploaded file by loading its content, splitting the text,
    and creating a vectorstore for retrieval. 
    """
    if file is None:
        return None, None
    
    filePath = file.name if hasattr(file, 'name') else file
    
    loader = UnstructuredFileLoader(filePath)  # Handles multiple formats
    # Add error handling
    try:
        docs = loader.load()
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None
    # Experiment with different splitting strategies
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = textSplitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return docs, vectorstore

def processFile(file):
    """
    Single function to process the uploaded file, returning docs, vectorstore, and a status message.
    """
    docs, vectorstore = processUploadedFile(file)
    if docs is None or vectorstore is None:
        return None, None, "Error al procesar el archivo. Asegúrate de que el archivo es válido."
    return docs, vectorstore, "Archivo procesado exitosamente."

def generateBookSummary(docs):
    """
    Generate book summary using the tool with the provided documents.
    """
    return createBookSummaryTool.invoke({
        "llm": llm, 
        "text_splitter": textSplitter, 
        "docs": docs
    })

def chatbot(message, vectorstore):
    """
    Build the prompt with conversation history and file-based context, then stream the LLM response.
    """
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 5, 'score_threshold': 0.5}
    )
    
    relevantDocs = retriever.invoke(message)
    context = []
    for doc in relevantDocs:
        source = doc.metadata.get('source', 'unknown')
        content = f"Source: {source}\n{doc.page_content}"
        context.append(content)
    
    finalPrompt = """ Eres un experto asistente de libros. Usa este contexto:
    {context}

    Pregunta: {question}
    
    - Responde en 2-5 frases
    - Cita la fuente si es posible
    - Si no estas segure responde: 'No he podido encontrar informacion en el libro'
    Respuesta:"""

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
    # TODO: Debug this process properly and see if new_hsitory is needed or not
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
        for response_chunk in chatbot(message, vectorstore):
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
        browserState = gr.BrowserState(default_value=[], storage_key="chat_history")
        docsState = gr.State(None)
        vectorState = gr.State(None)
        
        # File upload component for user-provided book file
        fileUpload = gr.File(label="Sube el archivo", file_count="single")
        fileButton = gr.Button("Procesar Archivo")
        fileStatus = gr.Textbox(label="Estado de archivo", interactive=False)
        
        # Chatbot interface components
        chatbotComponent = gr.Chatbot(type="messages", height=600)
        textbox = gr.Textbox(placeholder="Escribe tu mensaje aquí...", container=False, scale=7)
        
        with gr.Row():
            submitBtn = gr.Button("Enviar")
            
        examples = gr.Examples(
            examples=[
                "Haz un resumen del contenido",
                "¿Cuál es el argumento principal?",
                "Entra en más detalles sobre "
            ],
            inputs=[textbox],
            label="Ejemplos:"
        )

        fileButton.click(
            fn=processFile,
            inputs=[fileUpload],
            outputs=[docsState, vectorState, fileStatus]
        )
        
        submitEvent = textbox.submit(
            fn=processMessage,
            inputs=[textbox, browserState, vectorState, docsState],
            outputs=[chatbotComponent, browserState, vectorState, docsState],
            show_progress="hidden"
        )
        
        submitEvent.then(lambda: gr.Textbox(value=""), None, [textbox])
        
        submitBtn.click(
            fn=processMessage,
            inputs=[textbox, browserState, vectorState, docsState],
            outputs=[chatbotComponent, browserState, vectorState, docsState],
            show_progress="hidden"
        ).then(lambda: gr.Textbox(value=""), None, [textbox])

    return demo

if __name__ == "__main__":
    demo = createInterface()
    demo.launch()
