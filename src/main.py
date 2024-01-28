import psycopg2
import os
from dotenv import load_dotenv
from langchain import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
import streamlit as st
from htmlTemplates import user_template, bot_template, css
from jinja2 import Template
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import textwrap
from PyPDF2 import PdfReader


def read_tasks():
    df = pd.read_csv("Input/tasks1.csv", sep=';')
    columns = ['uuid', 'app_code', 'store_id', 'org_code', 'assigned_user']
    df_template = df[columns]
    return df_template

def template(df):
    template_str = """
        {% for index, row in df.iterrows() %}
        {% if row['uuid'] is not none %}
            A tarefa com o id {{ row['uuid'] }} pertence à loja {{ row['store_id'] }}, {% if row['org_code'] is not none %} categoria {{ row['org_code'] }},{% else %} categoria não definida, {% endif %}{% if row['assigned_user'] is not none %}e está atribuído ao utilizador {{ row['assigned_user'] }}.{% else %}e não está atribuído a nenhum utilizador.{% endif %}
        {% else %}
            A tarefa com o id está nulo para a loja {{ row['store_id'] }}, {% if row['org_code'] is not none %} categoria {{ row['org_code'] }},{% else %} categoria não definida, {% endif %}{% if row['assigned_user'] is not none %}e está atribuído ao utilizador {{ row['assigned_user'] }}.{% else %}e não está atribuído a nenhum utilizador.{% endif %}
        {% endif %}
        {% endfor %}
    """
    final_template = Template(template_str)
    rendered_text = final_template.render(df=df)
    return rendered_text

def create_pdf(rendered_text):
    c = canvas.Canvas(filename='assets/output.pdf', pagesize=letter)
    width, height = letter  # Tamanho da página (letter)

    # Configuração de fonte e tamanho
    c.setFont("Helvetica", 10)

    # Inicialização das coordenadas para a primeira linha
    x = 30
    y = height - 50  # Inicia no topo da página e desce

    # Envolve o texto para que se ajuste à largura da página
    lines = textwrap.wrap(rendered_text, width=70)

    for line in lines:
        # Desenha cada linha no PDF
        c.drawString(x, y, line)
        y -= 9  # Move para baixo para a próxima linha

        # Se chegarmos ao final da página, cria uma nova página
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 50  # Reseta a posição Y para o topo da nova página

    c.save()

def get_pdf_text(pdf_path):
    text = ""

    # Open the PDF file in binary read mode
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"  # Added a newline character for better readability

    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path)

    chat = ChatOpenAI(temperature=0)

    st.set_page_config(page_title="Tlantic Chatbot", page_icon="assets/unnamed.jpg")
    st.write(css, unsafe_allow_html=True)

    tasks = read_tasks()
    rendered_template = template(tasks)
    create_pdf(rendered_template)
    raw_text = get_pdf_text("assets/output.pdf")
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    st.session_state.conversation = get_conversation_chain(vectorstore)


    if "conversation" not in st.session_state:
            st.session_state.conversation = None
    if "chat_history" not in st.session_state:
            st.session_state.chat_history = None

    st.header("How can I help you today?")
    with st.sidebar:
        user_question = st.text_input("Ask a question :")

    if user_question:
        handle_userinput(user_question)


if __name__ == '__main__':
    main()
