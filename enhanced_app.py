import os
from sqlalchemy import create_engine, MetaData
from llama_index import LLMPredictor, ServiceContext, SQLDatabase, VectorStoreIndex
from llama_index.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from langchain.chat_models import ChatOpenAI
import streamlit as st


def get_db_url(username, password, host, port, mydatabase):
    return f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{mydatabase}"


def load_table_schema_objs(metadata_obj, sql_database):
    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [SQLTableSchema(
        table_name=table_name) for table_name in metadata_obj.tables.keys()]
    return table_node_mapping, table_schema_objs


def get_default_llm_predictor():
    return LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106"))


# Initialize conversation history as a global variable
conversation_history = []


def main():
    st.title("Chat with Database")

    # File uploader in the sidebar on the left
    with st.sidebar:
        # Input for OpenAI API Key
        openai_api_key = st.text_input("OpenAI API Key", type="password")

        # Set OPENAI_API_KEY as an environment variable
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # User input for database details
        username = st.text_input("Enter your database username:")
        password = st.text_input(
            "Enter your database password:", type="password")
        host = st.text_input("Enter your database host:")
        port = st.text_input("Enter your database port:")
        mydatabase = st.text_input("Enter your database name:")

    # Proceed only if all database details are provided
    if username and password and host and port and mydatabase and openai_api_key:
        # Create engine with user-provided details
        db_url = get_db_url(username, password, host, port, mydatabase)
        engine = create_engine(db_url)

        # Load all table definitions
        metadata_obj = MetaData()
        metadata_obj.reflect(engine)

        # Load table schema objects
        table_node_mapping, table_schema_objs = load_table_schema_objs(
            metadata_obj, SQLDatabase(engine))

        # We dump the table schema information into a vector index.
        obj_index = ObjectIndex.from_objects(
            table_schema_objs, table_node_mapping, VectorStoreIndex)

        # LLMPredictor with user-provided OpenAI API key
        llm_predictor = LLMPredictor(llm=ChatOpenAI(
            api_key=openai_api_key, temperature=0, model_name="gpt-3.5-turbo-1106"))

        # Create ServiceContext
        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor)

        # Construct a SQLTableRetrieverQueryEngine
        query_engine = SQLTableRetrieverQueryEngine(
            SQLDatabase(engine),
            obj_index.as_retriever(similarity_top_k=1),
            service_context=service_context,
        )

        # Display chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.text_input("Ask your question?"):
            st.session_state.messages.append(
                {"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Query the assistant using the latest chat history
            result = query_engine.query(prompt)
            full_response = result.response
            sql_query = result.metadata.get('sql_query', "")

            # Display assistant response and SQL query in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown(full_response)

                if sql_query:
                    st.write("SQL Query:")
                    st.write(sql_query)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
