import os
import tempfile
import PyPDF2
import sys
import streamlit as st
import boto3
import botocore
from botocore.config import Config
from rhubarb import VideoAnalysis, DocAnalysis, LanguageModels, SystemPrompts
import uuid
import re
import json
import time
import datetime
import ffmpeg
from io import BytesIO, StringIO
from urllib.parse import urlparse
from langchain import hub
from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langchain.chains import LLMMathChain
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentExecutor
from langchain.agents import AgentType
from langchain.agents import ConversationalChatAgent
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_experimental.utilities.python import PythonREPL
from langchain_experimental.tools import PythonREPLTool

awsconfig = botocore.config.Config(
    retries = {"max_attempts": 1},
    region_name = 'us-east-1',
    tcp_keepalive = True,
    read_timeout = 3600,
    connect_timeout = 5,
)
region_name = 'us-east-1'
bucket_name = 'ENTER_YOUR_S3_BUCKET'

SYSTEM_PROMPT_ENGLISH = """<system_message>You are a helpful assistant. Please answer in Markdown (or HTML) and always start
                        the final answer with "Final Answer: ". Also repeat your last thought after "Final Answer: ".
                        Please also always show the path or the train of thought to the result in the final answer.
                        For calculations always use NumExpr. Please answer my questions as comprehensively as possible.
                        Go into detail on all aspects, elaborate your thought process and arguments, and support your statements
                        with examples, data, or other evidence if possible. I want to thoroughly understand the topic, so do not
                        hesitate to make your answers as extensive and complete as feasible. However, avoid repetitions or
                        digressing excessively from the core topic. Take a deep breath before answering. Correctness and quality
                        is more important than the speed of answering. Detect the language that the user is using. Make sure to
                        use the same language in your response. Do not mention the language explicitly.</system_message>"""

SYSTEM_PROMPT_GERMAN = """<system_message>Du bist ein hilfreicher Assistent. Bitte antworte in Markdown (oder HTML) 
                       und beginne deine endgültige Antwort immer mit "Final Answer: ". Wiederhole auch deinen letzten 
                       Gedanken nach "Final Answer: ". Bitte zeige in der endgültigen Antwort immer den Weg oder den 
                       Gedankengang zum Ergebnis. Verwende für Berechnungen immer NumExpr. Bitte beantworte meine Fragen 
                       so umfassend wie möglich. Gehe auf alle Aspekte detailliert ein, erläutere deinen Denkprozess und 
                       deine Argumente und unterstütze deine Aussagen wenn möglich mit Beispielen, Daten oder anderen 
                       Belegen. Ich möchte das Thema gründlich verstehen, zögere also nicht, deine Antworten so ausführlich 
                       und vollständig wie möglich zu gestalten. Vermeide jedoch Wiederholungen oder übermäßiges 
                       Abschweifen vom Kernthema. Atme tief durch, bevor du antwortest. Korrektheit und Qualität sind 
                       wichtiger als die Geschwindigkeit der Antwort. Erkenne die Sprache, die der Benutzer verwendet. 
                       Stelle sicher, dass du in deiner Antwort dieselbe Sprache verwendest. Erwähne die Sprache nicht 
                       explizit.</system_message>"""

@st.cache_data(ttl='12h')
def convert_mov_to_mp4(input_file):
    """
    Convert video files to MP4 format using moviepy and downscale to Full HD if resolution is bigger.
    Works with any video format supported by moviepy.
    """
    try:
        # Get file name and extension
        file_name = input_file.name
        file_extension = file_name.lower().split('.')[-1]
        # Import required libraries
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.video.fx.Resize import Resize
        import tempfile
        import os
        from io import BytesIO
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_input:
            temp_input.write(input_file.getvalue())
            temp_input_path = temp_input.name
            temp_output_path = os.path.splitext(temp_input_path)[0] + '_new_file.mp4'
            # Process the video using moviepy
            try:
                # Load the video clip
                video_clip = VideoFileClip(temp_input_path)
                # Check resolution and downscale if needed
                width, height = video_clip.size
                full_hd_width = 1920
                full_hd_height = 1080
                if width > full_hd_width or height > full_hd_height:
                    st.info(f"Downscaling video from {width}x{height} to Full HD (1920x1080)")
                    # Calculate the resize factor to maintain aspect ratio
                    resize_factor = min(full_hd_width / width, full_hd_height / height)
                    # Use the Resize class from moviepy.video.fx.Resize
                    resize_effect = Resize(resize_factor)
                    video_clip = resize_effect.apply(video_clip)
                # Write the output file
                video_clip.write_videofile(temp_output_path, codec='libx264', audio_codec='aac', fps=30)
                video_clip.close()
                # Read the output file
                with open(temp_output_path, 'rb') as output_file:
                    output_bytes = output_file.read()
                # Clean up temporary files
                os.unlink(temp_input_path)
                os.unlink(temp_output_path)
                # Create a BytesIO object with the converted data
                output_buffer = BytesIO(output_bytes)
                output_buffer.name = os.path.splitext(file_name)[0] + '_new_buffer.mp4'
                return True, output_buffer
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                # If video processing fails, return the original file
                return False, input_file
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return False, input_file

@st.cache_data(ttl='12h')
def upload_to_s3(upfile, bucket_name=bucket_name, s3_file_name=None):
    """Upload a file to an S3 bucket"""
    # Check if the file is a MOV file and convert if needed
    is_converted, file_to_upload = convert_mov_to_mp4(upfile)
    # If no specific S3 file name is provided, use the original filename
    if s3_file_name is None:
        s3_file_name = upfile.name
    # replace all non 0-9a-zA-Z with underscore
    s3_file_name = re.sub('[^0-9a-zA-Z]+', '_', s3_file_name)
    s3_file_name = s3_file_name + '.mp4'
    # create proper s3 link in video subdir and with uuid
    s3_file_name = 'video/' + str(uuid.uuid4()) + '_' + s3_file_name.replace(' ', '_')
    # Create S3 client
    # Note: AWS credentials should be configured via environment variables
    # or AWS configuration files for security
    s3_client = boto3.client('s3', config=awsconfig)
    try:
        # Upload the file
        s3_client.upload_fileobj(file_to_upload, bucket_name, s3_file_name)
        s3_uri = f"s3://{bucket_name}/{s3_file_name}"
        return True, s3_uri
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return False, None

@st.cache_data(ttl='12h')
def transcribe_audio_with_own_bucket(s3_url, language='de-DE', output_bucket=bucket_name):
    transcribe = boto3.client('transcribe', region_name='us-east-1')
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    uuiduuid4 = str(uuid.uuid4())
    job_name = f'language-video-audio_{timestamp}_{uuiduuid4}'
    response = transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        LanguageCode=language,
        MediaFormat='mp4',
        Media={
            'MediaFileUri': s3_url
        },
        OutputBucketName=output_bucket,  # Use your own bucket
        OutputKey=f'transcriptions/{job_name}.json'
    )
    return response, job_name

@st.cache_data(ttl='12h')
def get_transcription_result_safe(job_name, output_bucket=bucket_name, max_wait_time=600):
    transcribe = boto3.client('transcribe', region_name='us-east-1')
    s3 = boto3.client('s3', region_name='us-east-1')
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        try:
            response = transcribe.get_transcription_job(
                TranscriptionJobName=job_name
            )
            status = response['TranscriptionJob']['TranscriptionJobStatus']
            print(f"Job status: {status}")
            if status == 'COMPLETED':
                # Construct the expected S3 path
                expected_key = f'transcriptions/{job_name}.json'
                try:
                    # Try to get the object from your bucket
                    print(f"Accessing s3://{output_bucket}/{expected_key}")
                    transcript_response = s3.get_object(Bucket=output_bucket, Key=expected_key)
                    transcript_content = transcript_response['Body'].read().decode('utf-8')
                    transcript_json = json.loads(transcript_content)
                    return {
                        'status': 'COMPLETED',
                        'transcript': transcript_json['results']['transcripts'][0]['transcript'],
                        'full_results': transcript_json,
                        'job_details': response['TranscriptionJob']
                    }
                except s3.exceptions.NoSuchKey:
                    # If the expected key doesn't exist, try to find it
                    print("Expected key not found, searching for transcript file...")
                    # List objects in the transcriptions folder
                    list_response = s3.list_objects_v2(
                        Bucket=output_bucket,
                        Prefix='transcriptions/'
                    )
                    if 'Contents' in list_response:
                        for obj in list_response['Contents']:
                            if job_name in obj['Key'] and obj['Key'].endswith('.json'):
                                print(f"Found transcript at: {obj['Key']}")
                                transcript_response = s3.get_object(Bucket=output_bucket, Key=obj['Key'])
                                transcript_content = transcript_response['Body'].read().decode('utf-8')
                                transcript_json = json.loads(transcript_content)
                                return {
                                    'status': 'COMPLETED',
                                    'transcript': transcript_json['results']['transcripts'][0]['transcript'],
                                    'full_results': transcript_json,
                                    'job_details': response['TranscriptionJob']
                                }
                    # If still not found, return the original URI
                    transcript_uri = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
                    return {
                        'status': 'COMPLETED_URI_ONLY',
                        'transcript_uri': transcript_uri,
                        'message': 'Job completed but transcript not found in expected location',
                        'job_details': response['TranscriptionJob']
                    }
            elif status == 'FAILED':
                return {
                    'status': 'FAILED',
                    'error': response['TranscriptionJob'].get('FailureReason', 'Unknown error'),
                    'job_details': response['TranscriptionJob']
                }
        except Exception as e:
            print(f"Error checking job status: {e}")
            return {'status': 'ERROR', 'error': str(e)}
        time.sleep(10)
    return {
        'status': 'TIMEOUT',
        'error': f'Job did not complete within {max_wait_time} seconds'
    }

@st.cache_data(ttl='12h')
def transcribe_video(s3_url, language, output_bucket=bucket_name):
    try:
        print("Starting transcription job...")
        response, job_name = transcribe_audio_with_own_bucket(s3_url, language, output_bucket)
        print(f"Job started: {job_name}")
        
        print("Waiting for transcription to complete...")
        result = get_transcription_result_safe(job_name, output_bucket)
        
        if result['status'] == 'COMPLETED':
            print("Transcription completed successfully!")
            return result
        else:
            print(f"Issue with transcription: {result.get('error', result.get('message', 'Unknown error'))}")
            return result
            
    except Exception as e:
        print(f"Error in transcription workflow: {str(e)}")
        return {'status': 'ERROR', 'error': str(e)}

@st.cache_data(ttl='12h')
def invoke_lambda_with_s3_url(lambda_function_name, s3_url):
    """
    Invokes an AWS Lambda function with an S3 URL as a parameter,
    waits for execution, and returns the parsed response.
    Args:
        lambda_function_name (str): Name of the Lambda function to invoke
        s3_url (str): S3 URL to pass as a parameter to the Lambda function
        
    Returns:
        dict: Parsed response from the Lambda function
    """
    # Initialize the Lambda client with increased timeout
    lambda_config = Config(
        connect_timeout=5,  # Connection timeout in seconds
        read_timeout=600,   # Read timeout in seconds (5 minutes)
        retries={'max_attempts': 2}
    )
    # Initialize the Lambda client
    lambda_client = boto3.client('lambda',
                                 region_name='us-east-1',
                                 config=lambda_config
                                 )
    # Prepare the payload with the S3 URL
    payload = {
        "s3_url": s3_url
    }
    print(datetime.datetime.now())
    print(f"Invoking Lambda function '{lambda_function_name}' with S3 URL: {s3_url}")
    try:
        # Invoke the Lambda function synchronously (RequestResponse invocation type)
        response = lambda_client.invoke(
            FunctionName=lambda_function_name,
            InvocationType='RequestResponse',  # This makes the call synchronous
            LogType='Tail',
            Payload=json.dumps(payload)
        )
        # Check if the invocation was successful
        if response['StatusCode'] != 200:
            raise Exception(f"Lambda invocation failed with status code: {response['StatusCode']}")
        # Parse the Lambda function response
        response_payload = json.loads(response['Payload'].read().decode('utf-8'))
        # Check if Lambda function returned an error
        if 'FunctionError' in response:
            error_message = response_payload.get('errorMessage', 'Unknown error')
            raise Exception(f"Lambda function returned an error: {error_message}")
        print(datetime.datetime.now())
        print("Lambda function executed successfully")
        return response_payload
    except Exception as e:
        print(f"Error invoking Lambda function: {str(e)}")
        raise

@st.cache_data(ttl='1h')
def video_bda(s3_url):
    lambda_function_name = "arn:aws:lambda:us-east-1:__NUMBER__:function:__NAME__"  # Replace with your Lambda function name
    try:
        # Call the Lambda function and get the response
        result = invoke_lambda_with_s3_url(lambda_function_name, s3_url)
        # Process the result
        if isinstance(result, dict):
            if 'body' in result:
                # If body is a JSON string, parse it
                try:
                    body = result['body']
                    if isinstance(body, str):
                        body = json.loads(body)
                    return (json.dumps(body, indent=2))
                except json.JSONDecodeError:
                    st.error(f"Raw body content: {result['body']}")
    except Exception as e:
        st.error(f"Error in main function: {str(e)}")

def parse_s3_uri(uri):
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip('/')

@st.cache_data(ttl='12h')
def download_from_s3_to_memory(s3_url):
    s3_client = boto3.client('s3', config=awsconfig)
    try:
        # Create a IO object
        file_obj = BytesIO()
        # Parse S3 URI to get bucket and key
        bucket, key = parse_s3_uri(s3_url)
        # Download the file to the BytesIO object
        s3_client.download_fileobj(bucket, key, file_obj)
        # Reset the file pointer to the beginning
        file_obj.seek(0)
        return True, file_obj
    except Exception as e:
        # Handle the error
        st.error(f"Error downloading file: {str(e)}")
        return False, None

def extract_pdf_text(pdf_file):
    """
    Extract text from a PDF file and return it as a string.
    Args:
        pdf_file: A file-like object containing PDF data
    Returns:
        str: The extracted text from the PDF
    """
    pdf_text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        try:
            # Reset the file pointer to the beginning
            pdf_file.seek(0)
            # Write the file content to the temporary file
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name
        except Exception as e:
            return f"Error reading file: {str(e)}"
        try:
            # Extract text from the PDF
            with open(tmp_file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                total_pages = len(pdf_reader.pages)
                # Extract text from each page
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    extracted_text = page.extract_text()
                    if extracted_text:  # Check if text was extracted
                        pdf_text += extracted_text + "\n\n"
        except Exception as e:
            return f"Error extracting text from PDF: {str(e)}"
        finally:
            # Clean up the temporary file
            os.unlink(tmp_file_path)
    return pdf_text

# define function for bedrock
def initialize_bedrock_llm(session):
    bedrock = session.client('bedrock-runtime',
                             'us-east-1',
                             endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                             verify=False,
                             config=awsconfig)
    llm = ChatBedrockConverse(client=bedrock, 
                              model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                              temperature=0.1,
                              max_tokens=None)
    return llm

# Initalize boto3 and bedrock session
session = boto3.Session()
llm = initialize_bedrock_llm(session)

# numexpr evaluation
@tool
def ne_evaluate(math_str: str) -> str:
    """MathChain for things which can be calculated: input is a string expression which have to be valid Python expression with numpy constants (np)"""
    import numexpr as ne
    import numpy as np
    try:
        result = str(ne.evaluate(math_str))
    except Exception as e:
        result = str(e)
    return result

tools = [
    # Python REPL for things which can be calculated using Python
    # PythonREPLTool(),
    # MathChain for things which can be calculated
    Tool(
        name = "NumExpr",
        func = ne_evaluate,
        description = "useful for calculations",
    ),
]

# Langchain Agent
memory = ConversationBufferMemory()
agent = initialize_agent(
    tools = tools,
    llm = llm,
    agent = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory = memory,
    max_iterations = 10,
    handle_parsing_errors = True,
    verbose = True
    )

def run_my_langchain(query: str):
    # Variables for Bedrock API
    modelId = 'us.anthropic.claude-3-5-sonnet-20241022-v2:0'
    contentType = 'application/json'
    accept = 'application/json'
    # Messages
    messages = [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": query
          }
        ]
      }
    ]
    # Body
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 5000,
        "messages": messages
    })
    with no_ssl_verification():
        response = agent.invoke(input = body, return_only_outputs=False)
    return response

# Function to reset chat context
def reset_chat_context():
    # Reset all session state variables related to chat
    for key in ["message", "answer_claude", "my_input", "trans_text"]:
        if key in st.session_state:
            st.session_state[key] = "" if isinstance(st.session_state.get(key, ""), str) else []
    # Reset memory in the agent
    memory.clear()

st.set_page_config(page_title='Video Content Extraction Tool', layout='wide')
st.header('Video Content Extraction Tool')

with st.container():
    uploaded_file = st.file_uploader("**Choose a file**", accept_multiple_files=False, type=['mp4', 'mov'])

# Initialize session state variables at the beginning of your app
if "message" not in st.session_state:
    st.session_state.message = []
    
if "answer_claude" not in st.session_state:
    st.session_state.answer_claude = ""
    
if "my_input" not in st.session_state:
    st.session_state.my_input = ""
    
if "trans_text" not in st.session_state:
    st.session_state.trans_text = ""
    
if "video_summary" not in st.session_state:
    st.session_state.video_summary = None
    
if "video_transcript" not in st.session_state:
    st.session_state.video_transcript = None

if "video_transcript_german" not in st.session_state:
    st.session_state.video_transcript_german = None

if "video_chapters_summaries" not in st.session_state:
    st.session_state.video_chapters_summaries = ""

if "additional_context" not in st.session_state:
    st.session_state.additional_context = ""

# Initialize language preference
if "german" not in st.session_state:
    st.session_state.german = False

if "spanish" not in st.session_state:
    st.session_state.spanish = False

with st.sidebar:
    st.write('**Language Selection**')
    st.write('⚠️ Language changes result in a reset of previous interactions')
    # Store previous language state
    previous_german = st.session_state.get('german', False)
    previous_spanish = st.session_state.get('spanish', False)
    # Determine the current selection based on session state
    if previous_german:
        current_selection = "Deutsch / German"
    elif previous_spanish:
        current_selection = "Español / Spanish"
    else:
        current_selection = "English"
    selected_language = st.selectbox(
        "Select Language",
        options=["English", "Deutsch / German", "Español / Spanish"],
        index=["English", "Deutsch / German", "Español / Spanish"].index(current_selection)
    )
    # Update session state based on selection
    st.session_state.german = (selected_language == "Deutsch / German")
    st.session_state.spanish = (selected_language == "Español / Spanish")
    # Check if language changed and rerun if needed
    if (st.session_state.german != previous_german) or (st.session_state.spanish != previous_spanish):
        reset_chat_context()
        st.rerun()

# Track the current file to detect changes
if "current_file" not in st.session_state:
    st.session_state.current_file = None

with st.form("my_form"):
    with st.expander("**Open to enter additional context**"):
        if 'additional_context' not in locals() and 'additional_context' not in globals():
            additional_context = ""
        additional_context = st.text_area(
            "Additional Information",
            placeholder="Enter any additional context or background information.",
            value=additional_context,
            height=100
        )
        # Store the additional context in session state
        st.session_state.additional_context = additional_context
        # Example usage in a Streamlit app:
        uploaded_pdf = st.file_uploader("**Choose a PDF file for additional context**", accept_multiple_files=False, type=['pdf'])
        if uploaded_pdf is not None:
            pdf_content = extract_pdf_text(uploaded_pdf)
            # st.text_area("PDF Content", pdf_content, height=300)
            st.session_state.additional_context += pdf_content
    submitted = st.form_submit_button("Start analysis")
    if submitted is True and uploaded_file is None:
        st.error('Error: No file uploaded')
    if submitted is True and uploaded_file is not None:
        # Check if this is a new file
        if st.session_state.current_file != uploaded_file.name:
            # Reset chat context when a new file is uploaded
            reset_chat_context()
            # Update current file
            st.session_state.current_file = uploaded_file.name
        with st.spinner("Uploading to S3 and analyzing video (may take several minutes)..."):
            success, s3_uri = upload_to_s3(uploaded_file)
        if success:
            st.success(f"File successfully uploaded to S3 bucket: {s3_uri}")
            try:
                with st.spinner('Processing video (may take several minutes)...'):
                    result = json.loads(video_bda(s3_uri))
                    result_json_status, result_json_io = download_from_s3_to_memory(result['result'])
                    result_json_io.seek(0)  # Reset position again if needed
                    result_json = json.loads(result_json_io.getvalue().decode('utf-8'))
                    # Store summary and transcript in session state for context
                    st.session_state.video_summary = result_json['video']['summary']
                    st.session_state.video_transcript = result_json['video']['transcript']['representation']['text']
                    if st.session_state.german:
                        with st.spinner('Processing audio...'):
                            result_german = transcribe_video(s3_uri, 'de-DE')
                            if result_german['status'] == 'COMPLETED':
                                st.session_state.video_transcript_german = result_german['transcript']
                    if st.session_state.spanish:
                        with st.spinner('Processing audio...'):
                            result_spanish = transcribe_video(s3_uri, 'es-ES')
                            if result_spanish['status'] == 'COMPLETED':
                                st.session_state.video_transcript_spanish = result_spanish['transcript']
                    # Build a string with chapter content 
                    for chapter in result_json["chapters"]:
                        iabs = []
                        if chapter.get("iab_categories"):
                            for iab in chapter["iab_categories"]:
                                iabs.append(iab["category"])            
                        # Append formatted text instead of printing
                        st.session_state.video_chapters_summaries += f'[{chapter["start_timecode_smpte"]} - {chapter["end_timecode_smpte"]}] {", ".join(iabs)}\n'
                        st.session_state.video_chapters_summaries += f'{chapter["summary"]}\n'
                        st.session_state.video_chapters_summaries += '\n'
                    with st.expander('**Raw Input from Video**'):
                        st.write('**Summary**')
                        st.write(st.session_state.video_summary)
                        if st.session_state.german:
                            st.write('**Transcript (German)**')
                            st.write(st.session_state.video_transcript_german)
                        elif st.session_state.spanish:
                            st.write('**Transcript (Spanish)**')
                            st.write(st.session_state.video_transcript_spanish)
                        else:
                            st.write('**Transcript (English)**')
                            st.write(st.session_state.video_transcript)
                        st.write('**Chapter summaries**')
                        st.write(st.session_state.video_chapters_summaries)
                    with st.container():
                        gtt = "Bitte fasse den Videoinhalt zusammen. Beachte bitte auch das Audiotranskript und den weiteren Kontext."
                        ett = "Please summarize the video content. Also note the audio transcript and the context."
                        st.session_state.trans_text = gtt if st.session_state.german else ett
                        question = st.text_input(
                            "Chat Input",
                            placeholder="Please enter your question about the video...",
                            value=st.session_state.get("trans_text", "")
                        )
                        if question:
                            # Include video summary and transcript as context
                            video_context = ""
                            if st.session_state.get("video_summary"):
                                video_context += f"\n\nVideo Summary: {st.session_state.video_summary}"
                            if st.session_state.german:
                                if st.session_state.get("video_transcript_german"):
                                    video_context += f"\n\nVideo Transcript (German): {st.session_state.video_transcript_german}"
                            elif st.session_state.spanish:
                                if st.session_state.get("video_transcript_spanish"):
                                    video_context += f"\n\nVideo Transcript (Spanish): {st.session_state.video_transcript_spanish}"
                            else:
                                if st.session_state.get("video_transcript"):
                                    video_context += f"\n\nVideo Transcript: {st.session_state.video_transcript}"
                            if st.session_state.get("video_chapters_summaries"):
                                video_context += f"\n\nVideo Chapters: {st.session_state.video_chapters_summaries}"
                            if st.session_state.get("additional_context"):
                                additional_context_text = st.session_state.get("additional_context", "").strip()
                                if additional_context_text:
                                    additional_context_header = "Zusätzlicher Kontext:" if st.session_state.german else "Additional Context:"
                                    video_context += f"\n\n{additional_context_header} {additional_context_text}"
                            system_message = SYSTEM_PROMPT_GERMAN if st.session_state.german else SYSTEM_PROMPT_ENGLISH
                            st.session_state.my_input = f"{system_message} This is the video context: {video_context}. And here is the next question: {question}"
                            with st.spinner(text="In progress..."):
                                try:
                                    answer = run_my_langchain(st.session_state.my_input)
                                    # Process the answer with more specific replacements
                                    processed_answer = answer['output']
                                    for prefix in ["Final Answer:", "Answer:"]:
                                        processed_answer = processed_answer.replace(prefix, "")
                                    st.session_state.answer_claude = processed_answer
                                    # Update message history
                                    if "message" not in st.session_state:
                                        st.session_state.message = []
                                    st.session_state.message.append({"role": "human", "content": question.strip()})
                                    # Process the answer for display
                                    display_answer = st.session_state.answer_claude
                                    display_answer = display_answer.replace("Final Answer:", "Previous Final:")
                                    display_answer = display_answer.replace("Answer:", "Previous:")
                                    st.session_state.message.append({"role": "assistant", "content": display_answer})
                                except ConnectionError as conn_err:
                                    st.error(f"Connection error: {conn_err}")
                                except ValueError as val_err:
                                    st.error(f"Value error: {val_err}")
                                except Exception as e:
                                    st.error(f"An unexpected error occurred: {str(e)}")
                            # Display message history
                            if st.session_state.get("message") and len(st.session_state.message) > 0:
                                for m in st.session_state.message:
                                    with st.chat_message(m["role"]):
                                        st.markdown(m["content"], unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error reading video: {str(e)}")
        else:
            st.error("Failed to upload video to S3")
