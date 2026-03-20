##############################################################
# The Lambda code is built as a backend integration for an AWS 
# API Gateway. It reads the instruction fed by the user along 
# and generated an image based on user input.  
###############################################################

import json
import boto3
import botocore
from datetime import datetime
import base64

#AWS Lambda Handler code
def lambda_handler(event, context):

    event = json.loads(event['body'])
    message = event['message']

    #Initiate  bedrock runtime boto3 client
    bedrock = boto3.client("bedrock-runtime",region_name="us-west-2",config = botocore.config.Config(read_timeout=300, retries = {'max_attempts':3}))

    s3 = boto3.client('s3')

    payload = {
        "text_prompts":[{f"text":message}],
        "cfg_scale":10,
        "seed":0,
        "steps":100
    }

    #Invoke stable diffusion model for image generation
    response = bedrock.invoke_model(body=json.dumps(payload),modelId = 'stability.stable-diffusion-xl-v0',contentType = "application/json",accept = "application/json")

    response_body = json.loads(response.get("body").read())
    base_64_img_str = response_body["artifacts"][0].get("base64")
    image_content = base64.decodebytes(bytes(base_64_img_str,"utf-8"))

    bucket_name = 'bedrock-image-bucket'
    current_time = datetime.now().strftime('%H%M%S')
    s3_key = f"output-images/{current_time}.png"

    #Upload generated image to S3 bucket
    s3.put_object(Bucket = bucket_name, Key = s3_key, Body = image_content, ContentType = 'image/png')



    return {
        'statusCode': 200,
        'body': json.dumps('Image Saved to s3')
    }