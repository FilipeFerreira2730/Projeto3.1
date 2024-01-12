import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
from langchain import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
import streamlit as st
from htmlTemplates import user_template, bot_template



def read_tasks():
    df = pd.read_csv("Input/tasks.csv", sep=';')
    columns = ['uuid', 'app_code', 'store_id', 'org_code', 'assigned_user']
    df['assigned_user'] = df['assigned_user'].replace(99999999, 0)
    df['assigned_user'] = df['assigned_user'].fillna(0).astype(int)
    df_template = df[columns]
    return df_template


def template(df):
    sentences = []
    for index, row in df.iterrows():
        sentence = f"o id {row['uuid']} pertence à loja {row['store_id']}, categoria {row['org_code']} e está atribuído ao utilizador {row['assigned_user']}."
        sentences.append(sentence)
    return sentences


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
    print("chunk data")
    print(data)
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def get_vectorstore_from_data(data_vector_store):
    # Create a DataFrame from the list
    print("data_vector_store")
    print(data_vector_store)


    data = pd.DataFrame({'task_template': data_vector_store})
    print("data")
    print(data)

    print("data-task-template")
    # Ensure all values in the 'task_template' column are strings
    data['task_template'] = data['task_template'].astype(str)
    print(data['task_template'])

    # Extract the text data from the DataFrame
    data_chunks = data['task_template'].tolist()
    print(data_chunks)
    # Initialize embeddings (replace this with your actual setup)
    embeddings = OpenAIEmbeddings()

    # Create the vector store
    vectorstore = FAISS.from_texts(texts=data_chunks, embedding=embeddings)

    return vectorstore



def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(),
                                                               memory=memory)
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
    print("OLA")
    #df = read_tasks()
    print("OLE")
    #generated_rows = template(df)
    print("oi")
    #insert_into_database(generated_rows)
    print("AAAAA")

    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path)

    print(os.getenv("OPENAI_API_KEY"))

    st.set_page_config(page_title="Tlantic Chatbot", page_icon=":memo:")


    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("How can I help you today?")
    user_question = st.text_input("Ask me a question:")
    if user_question:
        df = read_tasks()
        # base de dados
        tasks = read_tasks_from_database()
        data_vector_store = chunk_data(tasks, 300)
        vectorstore = get_vectorstore_from_data(data_vector_store)
        st.session_state.conversation = get_conversation_chain(vectorstore)
        handle_userinput(user_question)



if __name__ == '__main__':
    main()
