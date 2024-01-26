import psycopg2
import os
from dotenv import load_dotenv
from langchain import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
import streamlit as st
from htmlTemplates import user_template, bot_template, css
from jinja2 import Template
import pandas as pd


def read_tasks():
    df = pd.read_csv("Input/tasks1.csv", sep=';')
    columns = ['uuid', 'app_code', 'store_id', 'org_code', 'assigned_user']
    df_template = df[columns]
    return df_template


def template(df):
    template_str = """
    {% for index, row in df.iterrows() %}
        o id {{ row['uuid'] }} pertence à loja {{ row['store_id'] }},
        {% if row['org_code'] is not nan %}
            categoria {{ row['org_code'] }},
        {% else %}
            categoria não definida,
        {% endif %}
        {% if row['assigned_user'] is not nan 0 %}
            e está atribuído ao utilizador {{ row['assigned_user'] }}.
        {% else %}
            e não está atribuído a nenhum utilizador.
        {% endif %}
    {% endfor %}
    """
    final_template = Template(template_str)
    rendered_text = final_template.render(df=df)
    return rendered_text


def is_database_empty():
    db_params = {
        "host": "db-template",
        "port": 5432,
        "user": "p3",
        "password": "p3",
        "database": "p3",
    }
    conn = psycopg2.connect(
        host=db_params['host'],
        port=db_params['port'],
        user=db_params['user'],
        password=db_params['password'],
        database=db_params['database']
    )

    cursor = conn.cursor()

    # Check if the tasks table is empty
    cursor.execute("SELECT COUNT(*) FROM public.task")
    count = cursor.fetchone()[0]

    # Close the connection
    conn.close()

    return count == 0


def insert_into_database(sentences):
    db_params = {
        "host": "db-template",
        "port": 5432,
        "user": "p3",
        "password": "p3",
        "database": "p3",
    }
    conn = psycopg2.connect(
        host=db_params['host'],
        port=db_params['port'],
        user=db_params['user'],
        password=db_params['password'],
        database=db_params['database']
    )
    cursor = conn.cursor()
    for sentence in sentences:
        query = f"INSERT INTO public.task (task_template) VALUES ('{sentence}')"
        cursor.execute(query)
    conn.commit()
    cursor.close()
    conn.close()


def read_tasks_from_database():
    db_params = {
        "host": "db-template",
        "port": 5432,
        "user": "p3",
        "password": "p3",
        "database": "p3",
    }
    conn = psycopg2.connect(
        host=db_params['host'],
        port=db_params['port'],
        user=db_params['user'],
        password=db_params['password'],
        database=db_params['database']
    )
    query = "SELECT task_template FROM public.task"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def chunk_data(data, chunk_size):
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def get_vectorstore_from_data(data_vector_store, chunk_size):
    data_chunks = chunk_data(data_vector_store, chunk_size)

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_texts(texts=[str(chunk) for chunk in data_chunks], embedding=embeddings)

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

    print(vectorstore.as_retriever())
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            print("input")
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            print("response")
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path)

    chat = ChatOpenAI(temperature=0)

    if is_database_empty():
        df = read_tasks()
        generated_rows = template(df)
        insert_into_database(generated_rows)

    st.set_page_config(page_title="Tlantic Chatbot", page_icon="assets/unnamed.jpg")
    st.write(css, unsafe_allow_html=True)

    tasks = read_tasks_from_database()
    data_vector_store = chunk_data(tasks, 5)
    vectorstore = get_vectorstore_from_data(data_vector_store, 5)
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
