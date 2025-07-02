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

awsregion = 'us-east-1'

awsconfig = botocore.config.Config(
    retries = {"mode": "adaptive"},
    region_name = awsregion,
    tcp_keepalive = True,
    read_timeout = 300,
    connect_timeout = 5,
)

st.set_page_config(page_title='Information Extraction', layout='wide')
st.header('Information Extraction Tool')

from PIL import Image

def jpg_to_pdf_pillow_bytesio(jpg_bytes, output_pdf=None):
    # Open the image from bytes
    image = Image.open(BytesIO(jpg_bytes))
    # Convert to RGB if the image is in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # Create a BytesIO object to store the PDF
    pdf_bytes = BytesIO()
    # Save as PDF to the BytesIO object
    image.save(pdf_bytes, "PDF", resolution=300.0)
    # Reset the pointer to the beginning of the BytesIO object
    pdf_bytes.seek(0)
    # If output_pdf is provided, write to file
    if output_pdf:
        with open(output_pdf, 'wb') as f:
            f.write(pdf_bytes.getvalue())
    return pdf_bytes

with st.container():
    uploaded_file = st.file_uploader("**Choose a file**", accept_multiple_files=False, type=['pdf', 'jpg', 'png'])
    # "Please output all document information. Please use also tables."
    myprompt = "Please provide a summary first. Double-check the context of all information. Do not discard any information. Extract all information including all values as a structured JSON object with key-value pairs. Include entities, dates, numerical data, relationships, and any other significant information present in the document. Organize related information into nested objects including numbered pages where appropriate for better clarity."
    question = st.text_area('Question or Task ', value=myprompt, placeholder='Enter question or task...')
    mymodel = st.selectbox('Select LLM', ('AWS NOVA PRO', 'Claude Sonnet 3.5 v2'))
    if mymodel == 'Claude Sonnet 3.5 v2':
        modelId = LanguageModels.CLAUDE_SONNET_V2
    elif mymodel == 'Claude Sonnet 3.7':
        modelId = LanguageModels.CLAUDE_SONNET_37
    elif mymodel == 'Claude Sonnet 4':
        modelId = LanguageModels.CLAUDE_SONNET_4
    else:
        modelId = LanguageModels.NOVA_PRO
        
with st.form("my_form"):
    submitted = st.form_submit_button("Start analysis")
    if submitted is True and uploaded_file is None:
        st.error('Error: No file uploaded')
    elif submitted is True and len(question)<1:
        st.error('Error: Question is empty')
    elif submitted is True and uploaded_file is not None and len(question)>0:
        if os.path.splitext(uploaded_file.name)[1].lower() in ['.jpg', '.jpeg']:
            new_file_path = uploaded_file.name + '.pdf'
            uploaded_file = jpg_to_pdf_pillow_bytesio(uploaded_file.getvalue())
            uploaded_file.name = new_file_path
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1].lower()) as tmp_file:
            try:
                # Reset the file pointer of the uploaded file to the beginning
                uploaded_file.seek(0)
                # Write the uploaded file content to the temporary file
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            except Exception as e:
                st.error("Error reading file: " + str(e))
            try:
                session = boto3.Session(region_name=awsregion)
                da = DocAnalysis(file_path=tmp_file_path,
                                 modelId=modelId,
                                 boto3_session=session,
                                 max_tokens=4096,
                                 temperature=0.0,
                                 enable_cri=True,
                                 sliding_window_overlap=2,
                                 system_prompt=SystemPrompts().SchemaSysPrompt)
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
                if "output" in result:
                    output_data = result["output"]
                    if "page" in output_data:
                        st.success("PDF processed successfully!")
                        # Create tabs for each page
                        tabs = st.tabs([f"Page {page['page']}" for page in output_data])
                        # Fill each tab with content
                        for i, page in enumerate(output_data):
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
