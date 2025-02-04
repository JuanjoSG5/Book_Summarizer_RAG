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
    model_name="deepseek/deepseek-r1",
    model_kwargs={
        "extra_headers": {
            "Helicone-Auth": f"Bearer " + getenv("HELICONE_API_KEY")
        }
    },
)

url = "sample.txt"
loader = TextLoader(url)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

def generate_book_summary():
    """Helper function to generate book summary using the tool"""
    return create_book_summary_tool()

def chatbot(message, previous_history):
    # Check if message contains summary keyword
    if "resumen" in message.lower():
        summary = generate_book_summary()
        return [(message, summary)]
    
    relevant_docs = vectorstore.similarity_search(message)
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    final_prompt = (
        "Eres un asistente experto en solucionar dudas sobre conceptos descritos en un libro. "
        "Utiliza el siguiente contexto para responder de forma breve y concisa. "
        "Si no encuentras la información, responde que no la conoces.\n\n"
        f"Contexto:\n{context_text}\n\n"
        f"Pregunta: {message}\n"
        "Respuesta:"
    )

    messages = []
    for user_msg, bot_response in previous_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_response})
    messages.append({"role": "user", "content": final_prompt})

    response = llm.stream(messages)
    partial_response = ""

    for chunk in response:
        if chunk and hasattr(chunk, "content"):
            content = chunk.content
            if content is not None:
                partial_response += content
                yield partial_response

def process_message(message, history):
    # Check for summary keyword first
    if "resumen" in message.lower():
        summary = generate_book_summary()
        updated_history = history + [(message, summary)]
        yield updated_history
        return

    # Show processing message immediately
    display_history = history + [(message, "⏳ Procesando...")]
    yield display_history
    
    # Generate actual response
    full_response = ""
    for response_chunk in chatbot(message, history):
        full_response = response_chunk
        display_history[-1] = (message, full_response)
        yield display_history

def create_summary(history):
    summary = generate_book_summary()
    return history + [("Generar resumen del libro", summary)]

def create_interface():
    with gr.Blocks(title="ChatBot RAG - Resumen de libros", theme="ocean") as demo:
        gr.Markdown("## Asistente virtual resumidor de libros.")
        
        chatbot_component = gr.Chatbot(height=400)
        textbox = gr.Textbox(placeholder="Escribe tu mensaje aquí...", container=False, scale=7)
        
        with gr.Row():
            submit_btn = gr.Button("Enviar")
            summary_btn = gr.Button("Generar Resumen Completo")
            
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
        submit_event = textbox.submit(
            fn=process_message,
            inputs=[textbox, chatbot_component],
            outputs=[chatbot_component],
            show_progress="hidden"
        )
        submit_event.then(lambda: gr.Textbox(value=""), None, [textbox])
        
        # Submit button handler
        submit_btn.click(
            fn=process_message,
            inputs=[textbox, chatbot_component],
            outputs=[chatbot_component],
            show_progress="hidden"
        ).then(lambda: gr.Textbox(value=""), None, [textbox])
        
        # Summary button handler
        summary_btn.click(
            fn=create_summary,
            inputs=[chatbot_component],
            outputs=[chatbot_component],
            show_progress="hidden"
        )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()