from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings
import boto3

AWS_REGION = "us-east-1"

warnings.filterwarnings("ignore")

bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

model = ChatBedrock(
    model_id="amazon.nova-lite-v1:0",
    client=bedrock
)

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", client=bedrock
)

question = "Who is the creater of Gone With the Wind?"

# data ingestion
loader = PyPDFLoader("../assets/books.pdf")
splitter = RecursiveCharacterTextSplitter(separators=[". \n"], chunk_size=200)
docs = loader.load()
splitted_docs = splitter.split_documents(docs)

# create vector store
vector_store = FAISS.from_documents(splitted_docs, bedrock_embeddings)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.2,  # If similarity < threshold → returns EMPTY list, do not hallucinate
        "k": 5
    }
)

results = retriever.invoke(question)

results_string = []
for result in results:
    results_string.append(result.page_content)


# build template:
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strict AI assistant.
            Only answer using the provided context. You can be a bit descriptive if the context in right.
            If the answer is not contained in the context, say:
            "I am sorry, I don't have enough information in the provided context."
            Context:
            {context}
            """,
        ),
        ("user", "{input}"),
    ]
)

chain = template.pipe(model)

response = chain.invoke({"input": question, "context": results_string})
print(response.content)
