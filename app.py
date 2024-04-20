# -*- coding: utf-8 -*-
"""Said Lfagrouche: RAG_QA_App_base_on_BROOKLYN_COLLEGE STUDENT_HANDBOOK_2023-2024_app
"""

import getpass
import gradio as gr
import os
import pprint
import sys

from gradio.themes.base import Base
from icecream import ic
from pymongo import MongoClient
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from weaviate.embedded import EmbeddedOptions

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch, Weaviate


# langchain imports
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# langchain_community imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.chat_models import ChatOpenAI


# Retrieve environment variables.
OPENAI_API_KEYY = os.getenv('OPENAI_API_KEY')
MONGO_URII = os.getenv('MONGO_URI')

# Append our directory path to the Python system path.
directory_path = "Data"

sys.path.append(directory_path)

# Print the updated system path to the console.
print("sys.path =", sys.path)

# Get all the filenames under our directory path.
my_pdfs = os.listdir(directory_path)
my_pdfs

# Connect to MongoDB Atlas cluster using the connection string.
cluster = MongoClient(MONGO_URII)

# Define the MongoDB database and collection name.
DB_NAME = "pdfs"
COLLECTION_NAME = "pdfs_collection"

# Connect to the specific collection in the database.
MONGODB_COLLECTION = cluster[DB_NAME][COLLECTION_NAME]

vector_search_index = "vector_index"

# Load the PDF.
loaders = []
for my_pdf in my_pdfs:
  my_pdf_path = os.path.join(directory_path, my_pdf)
  loaders.append(PyPDFLoader(my_pdf_path))

print("len(loaders) =", len(loaders))

loaders

# Load the PDF.
# data = [loader.load() for loader in loaders]

data = []
for loader in loaders:
  data.append(loader.load())

print("len(data) =", len(data), "\n")

# First PDF file.
data[0]

# Initialize the text splitter.
# Uses a text splitter to split the data into smaller documents.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
text_splitter

# docs = [text_splitter.split_documents(doc) for doc in data]

docs = []
for doc in data:
  chunk = text_splitter.split_documents(doc)
  docs.append(chunk)

# Debugging purposes.
# Print the number of total documents to be stored in the vector database.
total = 0
for i in range(len(docs)):
  if i == len(docs) - 1:
    print(len(docs[i]), end="")
  else:
    print(len(docs[i]), "+ " ,end="")
  total += len(docs[i])
print(" =", total, " total documents\n")

# Print the first document.
print(docs[0], "\n\n\n")

# Print the total number of PDF files.
# docs is a list of lists where each list stores all the documents for one PDF file.
print(len(docs))

docs

# Merge the documents to be embededed and store them in the vector database.
merged_documents = []

for doc in docs:
  merged_documents.extend(doc)

# Print the merged list of all the documents.
print("len(merged_documents) =", len(merged_documents))
print(merged_documents)

# Hugging Face model for embeddings.
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
)

import weaviate
from weaviate.embedded import EmbeddedOptions

client = weaviate.Client(
    embedded_options=EmbeddedOptions()
)

vector_search = Weaviate.from_documents(
    client = client,
    documents = merged_documents,
    embedding = OpenAIEmbeddings(),
    by_text = False
)
# At this point, 'docs' are split and indexed in Weaviate, enabling text search capabilities.

# Semantic Search.
# query = "When is the spring recess at The City College of New York for Spring 2024?"
query = "Who is Prof. Langsam"
results = vector_search.similarity_search(query=query, k=10) # 10 most similar documents.

print("\n")
pprint.pprint(results)
# ic(results) # Debugging purposes.

# Semantic Search with Score.
# query = "When is the spring recess at The City College of New York for Spring 2024?"
query = "Is there a pool in campus?"
results = vector_search.similarity_search_with_score(
   query = query, k = 10 # 10 most similar documents.
)

pprint.pprint(results)
# ic(results) # Debugging purposes.

# Filter on metadata.
# Semantic search with filtering.
query = "Where is Data tools and algorithm exam taken?"

results = vector_search.similarity_search_with_score(
   query = query,
   k = 10, # 10 most similar documents.
   pre_filter = { "page": { "$eq": 1 } } # Filtering on the page number.
)

pprint.pprint(results)
# ic(results) # Debugging purposes.

# Instantiate Weaviate Vector Search as a retriever
retriever = vector_search.as_retriever(
   search_type = "similarity", # similarity, mmr, similarity_score_threshold. https://api.python.langchain.com/en/latest/vectorstores/langchain_core.vectorstores.VectorStore.html#langchain_core.vectorstores.VectorStore.as_retriever
   search_kwargs = {"k": 5, "score_threshold": 0.89}
)

# Define a prompt template.
# Define a LangChain prompt template to instruct the LLM to use our documents as the context.
# LangChain passes these documents to the {context} input variable and the user's query to the {question} variable.
template = """
Use the following pieces of context to answer the question at the end.
If you do not know the answer, just say that you do not know, do not try to make up an answer.

{context}

Question: {question}
"""

custom_rag_prompt = PromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2) # Increasing the temperature, the model becomes more creative and takes longer for inference.

# Input : docs (list of documents)
# Output: A single string that concatenates the page_content of each document in the list, separated by two newline characters.
def format_docs(docs):
   return "\n\n".join(doc.page_content for doc in docs)

# Regular chain format is defined as: chain = context_setup | prompt_template | model | output_parser

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}  # Setup the context and question for the chain
    | custom_rag_prompt  # Apply a custom prompt template to format the input for the LLM
    | llm  # Process the formatted input through a language model (LLM)
    | StrOutputParser()  # Parse the LLM's output into a structured format
)

# Prompt the chain.
query = "Where is Student Center located at?"
answer = rag_chain.invoke(query)

print("\nQuestion: " + query)
print("Answer: " + answer)

# Return the source documents
documents = retriever.get_relevant_documents(query)

print("\nSource documents:")
pprint.pprint(documents)

# Input : query.
# Output: answer.
def get_response(query):
  return rag_chain.invoke(query)

# Gradio application.
with gr.Blocks(theme=Base(), title="RAG QA App base on BROOKLYN COLLEGE STUDENT HANDBOOK 2023 - 2024, Weaviate As The Vector Database, and Gradio") as demo:
    gr.Markdown(
        """
        # RAG QA App base on BROOKLYN COLLEGE STUDENT HANDBOOK 2023 - 2024, Weaviate As The Vector Database, and Gradio
        """)
    textbox = gr.Textbox(label="Question:")
    with gr.Row():
        button = gr.Button("Submit", variant="primary")
    with gr.Column():
        output1 = gr.Textbox(lines=1, max_lines=10, label="Answer:")


# Call get_response function upon clicking the Submit button.
    button.click(get_response, textbox, outputs=[output1])

demo.launch(share=True)



