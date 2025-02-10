# ChatBot RAG - Resumidor de archivos

## Introducción

Este proyecto es una aplicación basada en Python que permite interactuar con un asistente virtual para resumir libros y responder preguntas sobre su contenido. La aplicación integra diversas herramientas y frameworks como **Gradio**, **LangChain**, y **OpenAI** para proveer una experiencia de consulta y resumen interactiva.

## Funcionalidades Básicas

### Carga y Procesamiento de Archivos

La aplicación permite subir archivos de libros en formato TXT y PDF. Dependiendo del tipo de archivo, se utiliza un loader específico:

- **TXT**: Se usa el TextLoader para cargar y procesar el contenido.
- **PDF**: Se usa el PyPDFLoader para extraer el contenido del archivo PDF.
  
### División de Texto:

Utiliza el `RecursiveCharacterTextSplitter` para dividir el contenido del libro en fragmentos manejables, optimizando la consulta y el resumen de información.

### Generación de Vectorstore:

Mediante el uso de Chroma, se genera un vectorstore que facilita la búsqueda de similitudes y la recuperación de información relevante.

### Interfaz de Chatbot:

La aplicación cuenta con una interfaz interactiva basada en Gradio que permite a los usuarios hacer preguntas y recibir respuestas en tiempo real, aunque en la interaccion actual no esta impelmentada la funcion del historial es algo que implementare en el futuro. El asistente utiliza un modelo de lenguaje de OpenAI para responder consultas basadas en el contenido del libro.

### Resumen Automático:
Si el usuario incluye la palabra "resumen" en su consulta, la aplicación genera automáticamente un resumen del contenido del libro mediante una herramienta específica integrada en el sistema.

## Uso de la Aplicación
Subir el Archivo del Libro:
Utiliza el componente de carga de archivos de la interfaz para seleccionar un libro en formato TXT o PDF.

Procesamiento del Archivo:
Al hacer clic en el botón "Procesar Archivo", la aplicación carga el contenido, lo divide en fragmentos y crea un vectorstore para facilitar la búsqueda de información.

Tiempo de procesamiento:
Archivos TXT: Aproximadamente 1 minuto.
Archivos PDF: Aproximadamente 2 minutos.
Interacción con el Chatbot:
Una vez procesado el archivo, puedes interactuar con el asistente escribiendo preguntas o solicitando un resumen del contenido. El chatbot utiliza el contexto extraído del archivo para proporcionar respuestas breves y concisas.

Tiempos de Procesamiento
Archivos TXT: El procesamiento completo, desde la carga hasta la creación del vectorstore, demora alrededor de 1 minuto.
Archivos PDF: Debido a la complejidad en la extracción y procesamiento del contenido, el tiempo aproximado es de 2 minutos.
Conclusión
Este proyecto facilita la interacción con un asistente virtual que ayuda a comprender y resumir libros de manera eficiente. Gracias a la integración de herramientas modernas de procesamiento de lenguaje y a una interfaz amigable, los usuarios pueden obtener respuestas rápidas y precisas sobre el contenido de sus documentos.
