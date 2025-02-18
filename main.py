
from os import getenv
from dotenv import load_dotenv
import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tool import createBookSummaryTool
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.prompts import PromptTemplate  # Importa PromptTemplate
from langchain_core.output_parsers import StrOutputParser #Import output parser

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

def processMessage(message, history, vectorstore, docs):
    """
    Process incoming messages. If the message contains "resumen", generate a book summary;
    otherwise, stream the chatbot response using the uploaded file’s vectorstore.
    """
    # TODO: Debug this process properly and see if new_history is needed or not
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

# Función principal del chatbot (adaptada para gr.ChatInterface)
def chatbot(message, history, vectorstore=None, docs=None):
    """Función principal para interactuar con el chatbot."""

    if message.lower().startswith("resumen") or "resumen" in message.lower():
        if not docs:
             yield "Por favor, sube un archivo primero."
             return
        summary = createBookSummaryTool(llm, textSplitter, docs)
        yield summary
        return


    if not vectorstore:
        yield "Por favor, sube un archivo y procésalo primero."
        return

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 10, 'lambda_mult': 0.7})
    relevant_docs = retriever.invoke(message)
    context = "\n\n".join([f"Fuente: {doc.metadata.get('source', 'desconocida')}\n{doc.page_content}" for doc in relevant_docs])

    prompt_template = PromptTemplate.from_template(
        """Eres un experto asistente de documentos. Usa este contexto:
        {context}

        Pregunta: {question}

        - Responde en 2-5 frases.
        - Si no estás seguro, responde: 'No he podido encontrar información en el documento.'
        Respuesta:"""
    )

    # Construye la cadena de conversación.  MUY IMPORTANTE:  Añade StrOutputParser()
    chain = prompt_template | llm | StrOutputParser()

    response = chain.stream({"context": context, "question": message})
    partial_response = ""
    for chunk in response:
       partial_response += chunk
       yield partial_response




# Interfaz de Gradio (usando gr.ChatInterface)
# Interfaz de Gradio (usando gr.ChatInterface)
def createInterface():
    with gr.Blocks(title="ChatBot RAG - Resumen de libros", theme="ocean") as demo:
        gr.Markdown("## Asistente virtual resumidor de libros")

        # Estados para el vectorstore y los documentos.
        docs_state = gr.State(None)
        vectorstore_state = gr.State(None)

        # Subida de archivos y botón de procesamiento
        with gr.Column():
            file_upload = gr.File(label="Sube el archivo", file_count="single")
            process_button = gr.Button("Procesar Archivo")
        file_status = gr.Textbox(label="Estado del archivo", interactive=False) #Para mostrar si se cargo correctamente

        # ChatInterface
        chat_interface = gr.ChatInterface(
            chatbot,
            additional_inputs=[vectorstore_state, docs_state], # Pasa los estados como entradas adicionales
            chatbot=gr.Chatbot(height=400),
            textbox=gr.Textbox(placeholder="Escribe tu pregunta o 'resumen'...", container=False, scale=7),
            examples=[
                ["Haz un resumen del contenido", None, None],  #  Ejemplo con entradas adicionales
                ["¿Cuál es el argumento principal?", None, None], #  Ejemplo con entradas adicionales
                ["Entra en más detalles sobre...", None, None],  #  Ejemplo con entradas adicionales
            ],
        )

        # Manejadores de eventos
        process_button.click(
            processFile,
            inputs=[file_upload],
            outputs=[docs_state, vectorstore_state, file_status]
        )


    return demo

if __name__ == "__main__":
    demo = createInterface()
    demo.launch()
