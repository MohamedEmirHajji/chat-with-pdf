import gradio as gr
import fitz
import ollama
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


available_models = ["llama3.2", "deepseek-r1"]
chat_history = []
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
database_name = 'vector_db'
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
vectorstore = None
llm = None
retriever = None
conversation_chain = None


class PdfDocument:
    def __init__(self, file_binary, database_name, embedding_model):
        self.file_binary = file_binary
        self.database_name = database_name
        self.embedding_model = embedding_model
        self.content = ''

    def __parse_pages(self):
        pages = []
        document = fitz.open(stream=self.file_binary)
        for page_number in range(document.page_count):
            page = document.load_page(page_number)
            pages.append(Document(page_content=page.get_text()))
        return pages

    def load_content(self):
        document = fitz.open(stream=self.file_binary)
        if self.content == '':
            for page_number in range(document.page_count):
                page = document.load_page(page_number)
                self.content += ' ' + page.get_text()
        return self.content

    def save(self):
        Chroma(persist_directory=self.database_name).delete_collection()
        parsed_pages = self.__parse_pages()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(parsed_pages)
        return Chroma.from_documents(documents=chunks, embedding=self.embedding_model, persist_directory=self.database_name)


def init_rag(pdf_document, model):
    global memory
    global vectorstore
    global llm
    global retriever
    global conversation_chain

    if vectorstore is None:
        vectorstore = pdf_document.save()
        llm = Ollama(model=model)
        retriever = vectorstore.as_retriever()
        conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)


def chat(message, file, model, enable_rag):
    global chat_history
    global conversation_chain

    pdf_document = PdfDocument(file, database_name, embedding_model)

    if enable_rag:
        init_rag(pdf_document, model)
        response = conversation_chain.invoke({"question": message})
        chat_history.append([message, response['answer']])
    else:
        messages = [
                {"role": "system", "content": "You are a smart assistant capable of summarizing long textual content, "
                                              "answering question by extracting useful content from the provided content or "
                                              "helping the user to memorize the document by asking questions related to "
                                              "the document and giving a rating on the answers. Here is the document: "
                                              "{document_content}".format(document_content=pdf_document.load_content())},
                {"role": "user", "content": "Process the provided document."}
            ]

        for user_message, assistant_message in chat_history:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_message})

        messages.append({"role": "user", "content": message})
        response = ollama.chat(model=model, messages=messages, think=False)
        chat_history.append([message, response.message.content])

    return chat_history


with gr.Blocks(title="Chat with PDF") as demo:
    gr.Label("💬 Chat with PDF")
    with gr.Row():
        output_textbox = gr.Chatbot(height=700)
        with gr.Column():
            file_input = gr.File(label="Drop your PDF here", type="binary", file_types=["pdf"])
            text_input = gr.Textbox(label="Ask anything", lines=4)
            model_input = gr.Dropdown(label="Choose a model", choices=available_models, value=available_models[0],
                                      interactive=True)
            enable_rag_input = gr.Checkbox(label="Enable RAG Mode", value=False)
            send_button = gr.Button("Send")
            send_button.click(chat, inputs=[text_input, file_input, model_input, enable_rag_input], outputs=[output_textbox])

demo.launch(debug=False)
