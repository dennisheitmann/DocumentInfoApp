import os
import sys
import streamlit as st
import boto3
import botocore
from rhubarb import VideoAnalysis, DocAnalysis, LanguageModels, SystemPrompts
import uuid

awsconfig = botocore.config.Config(
    retries = {"mode": "adaptive"},
    region_name = 'us-east-1',
    tcp_keepalive = True,
    read_timeout = 90,
    connect_timeout = 5,
)
region_name = 'us-east-1'
bucket_name = 'ENTER_YOUR_S3_BUCKET'

def upload_to_s3(upfile, bucket_name=bucket_name, s3_file_name=None):
    """Upload a file to an S3 bucket"""
    # If no specific S3 file name is provided, use the original filename
    if s3_file_name is None:
        s3_file_name = upfile.name
    s3_file_name = 'video/' + str(uuid.uuid4()) + '_' + s3_file_name.replace(' ', '_')
    # Create S3 client
    # Note: AWS credentials should be configured via environment variables
    # or AWS configuration files for security
    s3_client = boto3.client('s3', config=awsconfig)
    try:
        # Upload the file
        s3_client.upload_fileobj(upfile, bucket_name, s3_file_name)
        s3_uri = f"s3://{bucket_name}/{s3_file_name}"
        return True, s3_uri
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return False, None

st.set_page_config(page_title='Video Content Extraction Tool', layout='wide')
st.header('Video Content Extraction Tool')

with st.container():
    uploaded_file = st.file_uploader("**Choose a file**", accept_multiple_files=False, type=['mp4'])
    # "Please output all document information. Please use also tables."
    question = st.text_area('Question or Task ', value='Please provide a comprehensive summary of the video that includes: A brief overview of the main topic/purpose; Key points organized chronologically or by themes; Important timestamps or chapter markers if available; Any significant conclusions or takeaways; The structure should be clear with headings and bullet points for easy scanning.', placeholder='Enter question or task...') 
    
with st.form("my_form"):
    submitted = st.form_submit_button("Start analysis")
    if submitted is True and uploaded_file is None:
        st.error('Error: No file uploaded')
    if submitted is True and len(question)<1:
        st.error('Error: Question is empty')
    if submitted is True and uploaded_file is not None and len(question)>0:
        with st.spinner("Uploading to S3 and analyzing video..."):
            success, s3_uri = upload_to_s3(uploaded_file)
            if success:
                st.success(f"File successfully uploaded to S3 bucket: {s3_uri}")
                try:
                    session = boto3.Session(region_name=region_name)
                    da = VideoAnalysis(file_path=s3_uri,
                                       modelId=LanguageModels.NOVA_PRO,
                                       boto3_session=session,
                                       max_tokens=4096,
                                       temperature=0.0,
                                       system_prompt=SystemPrompts().SummarySysPrompt)
                    rresponse = da.run(message=question)
                except Exception as e:
                    st.error("Error reading video: " + str(e))
                try:
                    # Parse the response
                    # Assuming response is a JSON string or dictionary with the structure you provided
                    if isinstance(rresponse, str):
                        result = json.loads(rresponse)
                    else:
                        result = rresponse
                    st.write("**Content:**")
                    st.write(result["output"])
                    # Display token usage if available
                    st.divider()
                    if "token_usage" in result:
                        with st.expander("Token Usage"):
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Input Tokens", result["token_usage"]["input_tokens"])
                            col2.metric("Output Tokens", result["token_usage"]["output_tokens"])
                            col3.metric("Total Tokens", result["token_usage"]["total_tokens"])
                except Exception as e:
                    st.write(rresponse)
                    st.error("Response error: " + str(e))
            else:
                st.error("Failed to upload video to S3")
