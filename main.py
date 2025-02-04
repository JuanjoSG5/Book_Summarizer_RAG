from os import getenv
from dotenv import load_dotenv
import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tool import create_book_summary_tool

load_dotenv()

llm = ChatOpenAI(
    openai_api_key=getenv("OPENROUTER_API_KEY"),
    openai_api_base=getenv("OPENROUTER_BASE_URL"),
    model_name="google/gemini-flash-1.5-8b",
    model_kwargs={
        "extra_headers": {
            "Helicone-Auth": f"Bearer " + getenv("HELICONE_API_KEY")
        }
    },
)

url = "sample.txt"
loader = TextLoader(url)
docs = loader.load()

textSplitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=1000)
splits = textSplitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

def generateBookSummary():
    """Helper function to generate book summary using the tool"""
    return create_book_summary_tool.invoke({"llm":llm, "text_splitter":textSplitter, "docs":docs})

def chatbot(message, previousHistory):
    # Check if message contains summary keyword
    if "resumen" in message.lower():
        summary = generateBookSummary()
        return [(message, summary)]
    
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

    messages = []
    for user_msg, botResponse in previousHistory:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": botResponse})
    messages.append({"role": "user", "content": finalPrompt})

    response = llm.stream(messages)
    partialResponse = ""

    for chunk in response:
        if chunk and hasattr(chunk, "content"):
            content = chunk.content
            if content is not None:
                partialResponse += content
                yield partialResponse

def processMessage(message, history):
    displayHistory = history + [(message, "⏳ Procesando...")]
    yield displayHistory
    
    # Check for summary keyword first
    if "resumen" in message.lower():
        summary = generateBookSummary()
        updatedHistory = history + [(message, summary.content)]
        yield updatedHistory
    else:
        full_response = ""
        for response_chunk in chatbot(message, history):
            full_response = response_chunk
            displayHistory[-1] = (message, full_response)
            yield displayHistory

def createSummary(history):
    summary = generateBookSummary()
    return history + [("Generar resumen del libro", summary)]

def createInterface():
    with gr.Blocks(title="ChatBot RAG - Resumen de libros", theme="ocean") as demo:
        gr.Markdown("## Asistente virtual resumidor de libros.")
        
        chatbotComponent = gr.Chatbot(height=400)
        textbox = gr.Textbox(placeholder="Escribe tu mensaje aquí...", container=False, scale=7)
        
        with gr.Row():
            submitBtn = gr.Button("Enviar")
            # TODO: remove this button
            summaryBtn = gr.Button("Generar Resumen Completo")
            
        examples = gr.Examples(
            examples=[
                "¿De qué trata el libro?",
                "¿Cuál es el argumento principal?",
                "Haz un resumen del contenido",
                "¿Qué temas principales se abordan?"
            ],
            inputs=[textbox],
            label="Ejemplos:"
        )

        # Textbox submit handler
        submitEvent = textbox.submit(
            fn=processMessage,
            inputs=[textbox, chatbotComponent],
            outputs=[chatbotComponent],
            show_progress="hidden"
        )
        
        submitEvent.then(lambda: gr.Textbox(value=""), None, [textbox])
        
        # Submit button handler
        submitBtn.click(
            fn=processMessage,
            inputs=[textbox, chatbotComponent],
            outputs=[chatbotComponent],
            show_progress="hidden"
        ).then(lambda: gr.Textbox(value=""), None, [textbox])
        
        # Summary button handler
        summaryBtn.click(
            fn=createSummary,
            inputs=[chatbotComponent],
            outputs=[chatbotComponent],
            show_progress="hidden"
        )

    return demo

if __name__ == "__main__":
    demo = createInterface()
    demo.launch()