from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
import boto3

AWS_REGION = "us-east-1"

my_data = [
    "The weather is nice today.",
    "Last night's game ended in a tie.",
    "Anupam likes to eat pizza",
    "Anupam likes to eat pasta.",
]

question = "What does Anupam like to eat?"

bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

model = ChatBedrock(
    model_id="amazon.nova-lite-v1:0",
    client=bedrock
)

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", client=bedrock
)

# create vector store
vector_store = FAISS.from_texts(my_data, bedrock_embeddings)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}  # maybe we can add a score threshold here?
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
            "Answer the users question based on the following context: {context}",
        ),
        ("user", "{input}"),
    ]
)

chain = template.pipe(model)

response = chain.invoke({"input": question, "context": results_string})
print(response.content)
