import os
from dotenv import load_dotenv  # Load environment variables from a .env file

import textwrap  # (Optional) For formatting text if needed later

# Import various libraries used in the script
import langchain
import chromadb
import transformers
import torch
import requests
import json
import asyncio

# Import specific classes and functions from libraries
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM



# =============================================================================
# 1. Environment Setup and API Key Loading
# =============================================================================

# Load environment variables from the .env file (e.g., API keys)


# =============================================================================
# 6. Create the RetrievalQA Chain
# =============================================================================

# The RetrievalQA chain combines:
#   - The language model (model) to generate responses.
#   - A retriever (db.as_retriever) that fetches relevant document chunks based on the query.
#   - A prompt that provides instructions on how to answer the query.
def run(input):
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    load_dotenv()

    # =============================================================================
    # 2. Initialize the Language Model with Ollama
    # =============================================================================

    # Using an Arabic-capable model through Ollama
    model = OllamaLLM(model="llama3.1")

    # =============================================================================
    # 3. Document Preprocessing Function
    # =============================================================================

    def docs_preprocessing_helper(file):
        """
        Helper function to load and preprocess a CSV file containing data.
        
        This function performs two main tasks:
        1. Loads the CSV file using CSVLoader from LangChain.
        2. Splits the loaded documents into smaller text chunks using CharacterTextSplitter.
        
        Args:
            file (str): Path to the CSV file.
            
        Returns:
            list: A list of document chunks ready for embedding and indexing.
        
        Raises:
            TypeError: If the output is not in the expected dataframe/document format.
        """
        # Load the CSV file using LangChain's CSVLoader with UTF-8 encoding
        loader = CSVLoader(file, encoding="utf-8")
        docs = loader.load()
        
        # Create a text splitter that divides the documents into chunks of up to 1000 characters
        # with no overlapping text between chunks.
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = text_splitter.split_documents(docs)
        
        return docs

    # Preprocess the CSV file "faq.csv" and store the document chunks in 'docs'.
    docs = docs_preprocessing_helper('faq.csv')

    # =============================================================================
    # 4. Set Up the Embedding Function and Chroma Database
    # =============================================================================

    # Initialize the embedding function using a HuggingFace model with good Arabic support
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Create a vector store (ChromaDB) from the document chunks using the embedding function.
    # The Chroma database will be used to retrieve the most relevant documents based on the query.
    db = Chroma.from_documents(docs, embedding_function)

    # =============================================================================
    # 5. Define and Initialize the Prompt Template
    # =============================================================================

    # Define a prompt template that instructs the chatbot on how to answer FAQ questions in Arabic
    template = """أنت مساعد ذكي تجيب فقط على الأسئلة الموجودة في قاعدة بيانات الأسئلة المتداولة (FAQ) المرفقة.

    - استخدم الإجابات الموجودة في البيانات فقط.
    - لا تحاول تأليف أو تخمين أي معلومات خارج ما تم توفيره.
    - إذا كان السؤال لا يتطابق بشكل كافٍ مع الأسئلة الموجودة في البيانات، أجب برسالة: "عذراً، لا أملك إجابة على هذا السؤال، حاول مرة أخرى."
    - إذا كان هناك سؤال مشابه بشكل واضح في البيانات، استخدم إجابته كما هي أو بصياغة دقيقة وقريبة من الأصل.
    - لا تضف أي تفاصيل غير موجودة في البيانات.

    {context}

    سؤال المستخدم: {question}
    """
    # template = """أنت مساعد ذكي تجيب فقط على الأسئلة الموجودة في قاعدة بيانات الأسئلة المتداولة (FAQ) المرفقة.

    # الأسئلة والإجابات المتوفرة:
    # {context}

    # سؤال المستخدم: {question}

    # إذا لم تجد السؤال أو سؤال مشابه بشكل واضح، أجب بـ: "عذراً، لا أملك إجابة على هذا السؤال، حاول مرة أخرى."
    # """


    # Create the prompt template
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Create the RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )

    # Instead of calling with parameters directly, use the simpler .invoke method
    try:
        # Only pass the query/question parameter
        response = asyncio.run(chain.ainvoke({"query": input}))
        
        # Get the result from the response
        if isinstance(response, dict) and "result" in response:
            response_text = response["result"]
        else:
            response_text = str(response)
        
        # If no relevant information is found, return the default message in Arabic
        if not response_text or response_text.strip() == "":
            return "عذرا لا املك اجابه علي هذا السؤال, حاول مره اخري"
        
        return response_text
    except Exception as e:
        print(f"Error: {e}")
        return "حدث خطأ أثناء معالجة السؤال. يرجى المحاولة مرة أخرى."



