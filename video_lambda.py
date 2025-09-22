###############################################################################
# lambda: arn:aws:lambda:us-east-1:__ACCOUNT_ID__:function:__NAME__
# layer nv: arn:aws:lambda:us-east-1:__ACCOUNT_ID__:layer:__LAYER__:1
# python3.13
# timeout: 3m
# Permissions: AmazonBedrock, S3
# S3 bucket: __NAME__ 
###############################################################################
import json
import boto3
import os
import urllib.parse
import time
import uuid
import zipfile
from datetime import datetime
from urllib.parse import urlparse
import re

# Initialize Bedrock Data Automation client
current_region = 'us-east-1'
bda_client = boto3.client('bedrock-data-automation', region_name=current_region)
bda_runtime_client = boto3.client('bedrock-data-automation-runtime', region_name=current_region)
s3_client = boto3.client('s3', region_name=current_region)
account_id = '__ACCOUNT_ID__' # You can replace this with boto3.client('sts').get_caller_identity().get('Account')
project_name= "my_bda_video_project"
outputbucketpath = 'videooutput'

standard_output_config =  {
  "document": {
    "extraction": {
      "granularity": {"types": ["DOCUMENT","PAGE","ELEMENT"]},
      "boundingBox": {"state": "ENABLED"}
    },
    "generativeField": {"state": "ENABLED"},
    "outputFormat": {
      "textFormat": {"types": ["PLAIN_TEXT", "MARKDOWN", "HTML", "CSV"]},
      "additionalFileFormat": {"state": "ENABLED"}
    }
  },
  "image": {
    "extraction": {
      "category": {
        "state": "ENABLED",
        "types": ["TEXT_DETECTION","LOGOS"]
      },
      "boundingBox": {"state": "ENABLED"}
    },
    "generativeField": {
      "state": "ENABLED",
      "types": ["IMAGE_SUMMARY","IAB"]
    }
  },
  "video": {
    "extraction": {
        "category": {
            "state": "ENABLED",
            "types": ["TEXT_DETECTION", "TRANSCRIPT", "LOGOS"],
        },
        "boundingBox": {
            "state": "ENABLED",
        }
    },
    "generativeField": {
        "state": "ENABLED",
        "types": ["VIDEO_SUMMARY", "CHAPTER_SUMMARY", "IAB"],
    }
  },
  "audio": {
    "extraction": {
        "category": {
            "state": "ENABLED",
            "types": [
                "TRANSCRIPT"
            ]
        }
    },
    "generativeField": {
        "state": "ENABLED",
        "types": [
            "AUDIO_SUMMARY",
            "TOPIC_SUMMARY"
        ]
    }
  }
}

def wait_for_job_to_complete(invocationArn, max_attempts=120, delay=2):
    for attempt in range(max_attempts):
        response = bda_runtime_client.get_data_automation_status(
            invocationArn=invocationArn
        )
        status = response['status']
        if status not in ['InProgress']:
            return status
        time.sleep(delay)  # Wait for 2 seconds before checking again

def parse_s3_uri(uri):
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip('/')

def lambda_handler(event, context):
    rundatetimestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # Get the object details from the event
    try:
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'])
        s3_url = f"s3://{bucket}/{key}"
        print(f"Triggered by S3 event. Bucket: {bucket}, Key: {key}")
    except:
        # If not an S3 event, try to get s3_url from direct invocation
        s3_url = event.get('s3_url')
        if not s3_url:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing s3_url parameter or invalid S3 event'})
            }
        # Parse the S3 URL for direct invocation
        if not s3_url.startswith('s3://'):
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid S3 URL format'})
            }
        # Parse S3 URI to get bucket and key
        bucket, key = parse_s3_uri(s3_url)

    # Check if the file is in the videoinput folder and is a MP4/MOV
    if not(key.startswith('videoinput/') or key.startswith('video/')) or not(key.lower().endswith('mp4') or key.lower().endswith('mov')):
        print(f"Skipping file {key} as it's not a MOV/MP4 in the videoinput folder")
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid file type or folder'})
        }
    
    try:
        # delete project if it already exists
        projects_existing = [project for project in bda_client.list_data_automation_projects()["projects"] if project["projectName"] == project_name]
        print(f"projects_existing: {projects_existing}")
        if len(projects_existing) > 0:
            print(f"Deleting existing project: {projects_existing[0]}")
            bda_client.delete_data_automation_project(projectArn=projects_existing[0]["projectArn"])
            time.sleep(1) # nosemgrep

        response = bda_client.create_data_automation_project(
            projectName=project_name,
            projectDescription="BDA project to get extended standard output",
            projectStage='LIVE',
            standardOutputConfiguration=standard_output_config    
        )
        project_arn = response["projectArn"]
        print(f"project_arn: {project_arn}")
        time.sleep(1) # nosemgrep
        # Run BDA job
        input_s3 = 's3://' + bucket + '/' + key
        filename_clean = re.sub('[^0-9a-zA-Z]+', '_', os.path.basename(key).replace(' ', '_'))
        output_path = rundatetimestamp + '_' + filename_clean + '_' + str(uuid.uuid4()) + '_processed'
        output_s3 = 's3://' + bucket + '/' + outputbucketpath + '/' + output_path
        response = bda_runtime_client.invoke_data_automation_async(
            inputConfiguration={
                's3Uri': input_s3
            },
            outputConfiguration={
                's3Uri': output_s3
            },
            dataAutomationConfiguration={
                'dataAutomationProjectArn': project_arn,
                'stage': 'LIVE'
            },
            dataAutomationProfileArn = f'arn:aws:bedrock:{current_region}:{account_id}:data-automation-profile/us.data-automation-v1'
        )
        print(f"response: {response}")
        invocationArn = response['invocationArn']
        print(f"invocationArn: {invocationArn}")
        status_response = wait_for_job_to_complete(invocationArn=invocationArn)
        time.sleep(3)
        destination_key = ''
        if status_response == 'Success' or status_response == 'PROCESSED':
            time.sleep(3)
            # Extract the invocation ID from the ARN
            # ARN format: arn:aws:bedrock:region:account:data-automation-invocation/invocation-id
            invocation_id = invocationArn.split('/')[-1]
            # Construct the path to result.json
            file_key = f"{output_s3.rstrip('/')}/{invocation_id}/0/standard_output/0/result.json"
            print(f"Result file link: {file_key}")
        else:
            print (f"Error status: {status_response}")
        return {
            'statusCode': 200,
            'body': json.dumps({
                "status": f"Processed {key} with status: {status_response}",
                "input": input_s3,
                "output": output_s3,
                "result": file_key
            })
        }
    except Exception as e:
        print(f"Error processing {key}: {str(e)}")
        raise e
