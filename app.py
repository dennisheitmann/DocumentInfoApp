# Dennis Heitmann - 2025-06-05
import os
import sys
import streamlit as st
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
import re
import boto3
import botocore
from rhubarb import DocAnalysis, LanguageModels, SystemPrompts
import tempfile
import shutil

awsconfig = botocore.config.Config(
    retries = {"mode": "adaptive"},
    region_name = 'us-east-1',
    tcp_keepalive = True,
    read_timeout = 90,
    connect_timeout = 5,
)

st.set_page_config(page_title='Information Extraction', layout='wide')
st.header('Information Extraction Tool')

from PIL import Image

def jpg_to_pdf_pillow(jpg_path, pdf_path):
    # Open the image
    image = Image.open(jpg_path)
    # Convert to RGB if the image is in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # Save as PDF
    image.save(pdf_path, "PDF", resolution=300.0)

with st.container():
    uploaded_file = st.file_uploader("**Choose a file**", accept_multiple_files=False, type=['pdf', 'jpg'])
    # "Please output all document information. Please use also tables."
    question = st.text_area('Question or Task ', value='Please provide a concise summary of the document highlighting the main points. Then extract all relevant information as a structured JSON object with key-value pairs. Include important entities, dates, numerical data, relationships, and any other significant information present in the document. Organize related information into nested objects where appropriate for better clarity', placeholder='Enter question or task...') 
    bypage = st.toggle('Apply the question to each page individually.') 

with st.form("my_form"):
    submitted = st.form_submit_button("Start analysis")
    if submitted is True and uploaded_file is None:
        st.error('Error: No file uploaded')
    if submitted is True and len(question)<1:
        st.error('Error: Question is empty')
    if submitted is True and uploaded_file is not None and len(question)>0:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            try:
                # Reset the file pointer of the uploaded file to the beginning
                uploaded_file.seek(0)
                # Write the uploaded file content to the temporary file
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            except Exception as e:
                st.error("Error reading file: " + str(e))
            try:
                session = boto3.Session(region_name='us-east-1')
                if bypage:
                    da = DocAnalysis(file_path=tmp_file_path,
                                     modelId=LanguageModels.CLAUDE_SONNET_V2,
                                     max_tokens=4096,
                                     temperature=0.0,
                                     boto3_session=session)
                else:
                    da = DocAnalysis(file_path=tmp_file_path,
                                     modelId=LanguageModels.CLAUDE_SONNET_V2,
                                     boto3_session=session,
                                     max_tokens=4096,
                                     temperature=0.0,
                                     system_prompt=SystemPrompts().SummarySysPrompt)
                rresponse = da.run(message=question)
                os.unlink(tmp_file_path)
            except Exception as e:
                st.error("Error reading PDF file: " + str(e))
            try:
                # Parse the response
                # Assuming response is a JSON string or dictionary with the structure you provided
                if isinstance(rresponse, str):
                    result = json.loads(rresponse)
                else:
                    result = rresponse
                if bypage:
                    if "output" in result:
                        st.success("PDF processed successfully!")
                        
                        # Create tabs for each page
                        tabs = st.tabs([f"Page {page['page']}" for page in result["output"]])
                        
                        # Fill each tab with content
                        for i, page in enumerate(result["output"]):
                            with tabs[i]:
                                st.subheader(f"Page {page['page']}")
                                st.write(f"**Detected Languages:** {', '.join(page['detected_languages'])}")
                                st.write("**Content:**")
                                st.write(page['content'])
                else:
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
