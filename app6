import streamlit as st
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Function to handle CSV file upload and processing
def process_csv(file):
    if file is not None:
        df = pd.read_csv(file)
        return df
    return None

# Set up OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Function to initialize embeddings and FAISS database
def initialize_embeddings_and_db(file_path):
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(documents, embeddings)
    return db

# Function for similarity search
def retrieve_info(query, db):
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# Function to generate response
def generate_response(message, db, chain):
    best_practice = retrieve_info(message, db)
    response = chain.run(message=message, best_practice=best_practice)
    return response

# Main function
def main():
    st.set_page_config(
        page_title="Customer response generator", page_icon=":bird:"
    )

    st.header("Customer response generator :bird:")
    
    # File uploader for CSV file
    csv_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    # Process uploaded CSV file
    if csv_file is not None:
        df = process_csv(csv_file)
        file_path = "uploaded_file.csv"
        df.to_csv(file_path, index=False)
        
        # Initialize embeddings and database
        db = initialize_embeddings_and_db(file_path)
        
        # Allow user to input LLMChain setup text
        st.subheader("LLMChain & Prompts Setup")
        template = st.text_area(
            "LLMChain & Prompts Template",
            """
            You are a world class Property Administrator and a representative of Jettings Real Estate. 
            I will share a prospect's message with you and you will give me the best answer that 
            I should send to this prospect based on past best practices, 
            and you will follow ALL of the rules below:

            1/ Response should be very similar or even identical to the past best practices, 
            in terms of length, tone of voice, logical arguments and other details

            2/ If the best practices are irrelevant, then try to mimic the style of the best practices to the prospect's message

            Below is a message I received from the prospect:
            {message}

            Here is a list of best practices of how we normally respond to prospect in similar scenarios:
            {best_practice}

            Please write the best response that I should send to this prospect:
            """,
            height=400  # Set height to 400 pixels
        )
        
        prompt = PromptTemplate(
            input_variables=["message", "best_practice"],
            template=template
        )
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Allow user to enter customer message
        st.subheader("Customer Message")
        message = st.text_area("Enter customer message:")
        
        # Generate response when message is provided
        if message:
            st.write("Generating best practice message...")
            result = generate_response(message, db, chain)
            st.info(result)

# Execute the main function
if __name__ == '__main__':
    main()
