from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
import boto3

AWS_REGION = "us-east-1"

bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

model = ChatBedrock(
    model_id="amazon.nova-lite-v1:0",
    client=bedrock
)

#def invoke_model():
#    response = model.invoke("What is the highest mountain in the world?")
#    print(response)


def first_chain():
    prompt = ChatPromptTemplate.from_template(
        "Write a short, compelling product description for: {product_name}"
    )

    chain = prompt | model

    response = chain.invoke({"product_name": "bicycle"})
    print(response.content)

#invoke_model()
first_chain()