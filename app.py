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
import ast
import PyPDF2
import json
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

awsregion = 'us-east-1'

awsconfig = botocore.config.Config(
    retries = {"mode": "adaptive"},
    region_name = awsregion,
    tcp_keepalive = True,
    read_timeout = 300,
    connect_timeout = 5,
)

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

@st.cache_data(ttl='12h')
def da_run(tmp_file_path, modelId, pages_no, sysprompt, question):
    reset_chat_context()
    session = boto3.Session(region_name=awsregion)
    bedrock = session.client('bedrock-runtime', verify=False, config=awsconfig)
    da = DocAnalysis(file_path=tmp_file_path,
                     modelId=modelId,
                     boto3_session=session,
                     max_tokens=4096,
                     temperature=0.1,
                     enable_cri=True,
                     pages=pages_no,
                     sliding_window_overlap=0,
                     use_converse_api=True,
                     system_prompt=sysprompt)
    # Manually set the bedrock client
    da._bedrock_client = bedrock  # Note: accessing private attributes is generally not recommended
    question = question + f' The date of today: {datetime.now()}'
    return da.run(message=question)

# Function to reset chat context
def reset_chat_context():
    # Reset all session state variables related to chat
    for key in ["message", "answer_claude", "my_input", "output_data"]:
        if key in st.session_state:
            st.session_state[key] = "" if isinstance(st.session_state.get(key, ""), str) else []
    # Reset memory in the agent
    memory.clear()

st.set_page_config(page_title='Information Extraction', layout='wide')
st.header('Information Extraction Tool')

from PIL import Image, ExifTags

def img_to_pdf_pillow_bytesio(img_bytes, output_pdf=None):
    # Open the image from bytes
    image = Image.open(BytesIO(img_bytes))
    # Convert to RGB if the image is in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # Create a BytesIO object to store the PDF
    pdf_bytes = BytesIO()
    # Save as PDF to the BytesIO object
    image.save(pdf_bytes, "PDF", resolution=150.0)
    # Reset the pointer to the beginning of the BytesIO object
    pdf_bytes.seek(0)
    # If output_pdf is provided, write to file
    if output_pdf:
        with open(output_pdf, 'wb') as f:
            f.write(pdf_bytes.getvalue())
    return pdf_bytes

def img_to_png_pillow_bytesio(img_bytes, auto_rotate=True, max_width=2560, colors=256):
    # Print original size
    original_size = len(img_bytes)
    # Open the image from bytes
    image = Image.open(BytesIO(img_bytes))
    # Auto-rotate image based on EXIF data if enabled
    if auto_rotate:
        try:
            # Check if image has EXIF data
            if hasattr(image, '_getexif') and image._getexif() is not None:
                exif = dict((ExifTags.TAGS[k], v) for k, v in image._getexif().items()
                           if k in ExifTags.TAGS)
                # Check for orientation tag
                if 'Orientation' in exif:
                    orientation = exif['Orientation']
                    # Apply rotation based on EXIF orientation
                    if orientation == 2:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 3:
                        image = image.transpose(Image.ROTATE_180)
                    elif orientation == 4:
                        image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    elif orientation == 5:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
                    elif orientation == 6:
                        image = image.transpose(Image.ROTATE_270)
                    elif orientation == 7:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
                    elif orientation == 8:
                        image = image.transpose(Image.ROTATE_90)
                    else:
                        pass
                        #print("No EXIF rotation needed (orientation normal)")
                else:
                    #pass
                    print("No EXIF orientation data found")
            else:
                #pass
                print("No EXIF data found in image")
        except Exception as e:
            print(f"Error processing EXIF rotation: {str(e)}")
    # Resize image if it's larger than max_width
    if image.width > max_width:
        ratio = max_width / float(image.width)
        new_height = int(image.height * ratio)
        image = image.resize((max_width, new_height), Image.LANCZOS)
    # Handle alpha channel optimization
    has_alpha = image.mode in ('RGBA', 'LA')
    if has_alpha:
        # Check if alpha channel is actually being used meaningfully
        alpha = image.getchannel('A')
        if alpha.getextrema() == (255, 255):  # All pixels are fully opaque
            image = image.convert('RGB' if image.mode == 'RGBA' else 'L')
            has_alpha = False
        # Reduce colors if specified (great for PNG size reduction)
    if image.mode not in ('L', 'LA', 'P'):
        # Don't reduce if already grayscale or paletted
        if has_alpha:
            # Handle images with transparency
            # Split alpha channel
            rgb_image = image.convert('RGB')
            alpha_channel = image.getchannel('A')
            # Quantize the RGB part
            quantized = rgb_image.quantize(colors=colors, method=2, dither=Image.FLOYDSTEINBERG)
            # Convert back to RGBA and restore alpha
            image = quantized.convert('RGBA')
            image.putalpha(alpha_channel)
        else:
            image = image.quantize(colors=colors, method=2, dither=Image.FLOYDSTEINBERG)
    # Create a BytesIO object to store the PDF
    png_bytes = BytesIO()
    # Save as PNG to the BytesIO object
    image.save(png_bytes, "PNG", optimize=True, compress_level=9)
    # Reset the pointer to the beginning of the BytesIO object
    png_bytes.seek(0)
    # Get size
    png_bytes.seek(0)
    png_size = len(png_bytes.getvalue())
    # Print summary
    print(f"Original: {original_size / 1024:.2f} KB")
    print(f"Final: {png_size / 1024:.2f} KB ({(png_size/original_size)*100:.1f}% of original size)")
    png_bytes.seek(0)
    return png_bytes

with st.container():
    # Initialize session state variables at the beginning of your app
    if "message" not in st.session_state:
        st.session_state.message = []
    if "answer_claude" not in st.session_state:
        st.session_state.answer_claude = ""
    if "my_input" not in st.session_state:
        st.session_state.my_input = ""
    if "output_data" not in st.session_state:
        st.session_state.output_data = ""
    uploaded_file = st.file_uploader("**Choose a file**", accept_multiple_files=False, type=['pdf', 'jpg', 'png'], on_change=reset_chat_context)
    if uploaded_file is not None:
        if os.path.splitext(uploaded_file.name)[1].lower() in ['.jpg', '.jpeg']:
            new_file_path = uploaded_file.name + '.png'
            uploaded_file = img_to_png_pillow_bytesio(uploaded_file.getvalue())
            uploaded_file.name = new_file_path
            is_pdf = False
        elif os.path.splitext(uploaded_file.name)[1].lower() in ['.png']:
            is_pdf = False
        else:
            is_pdf = True
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1].lower()) as tmp_file:
            try:
                # Reset the file pointer of the uploaded file to the beginning
                uploaded_file.seek(0)
                # Write the uploaded file content to the temporary file
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            except Exception as e:
                st.error("Error reading file: " + str(e))
            if is_pdf:
                # Get PDF information
                with open(tmp_file_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    total_pages = len(pdf_reader.pages)
                    # Initialize an empty string to store the text
                    full_text = ""
                    # Extract text from each page and add it to full_text
                    for page_num in range(total_pages):
                        page = pdf_reader.pages[page_num]
                        full_text += page.extract_text()
                    # Display information to the user
                    st.write(f"Total pages in document: {total_pages}")
                    # Create page selection options
                    all_page_options = list(range(1, total_pages + 1))
                    # Create a radio button for selection mode
                    selection_mode = st.radio(
                        "Page selection mode:",
                        ["Process first 20 pages", "Select specific pages (max 20)"]
                    )
                    pages_no = []  # Initialize the pages list
                    if selection_mode == "Process first 20 pages":
                        # Set pages to [0] to indicate "process first 20 pages"
                        pages_no = [0]
                        st.info("The first 20 pages will be processed.")
                        
                        # Calculate how many pages will actually be processed
                        pages_to_process = min(20, total_pages)
                        st.write(f"Pages to be processed: 1 to {pages_to_process}")
                    else:  # Select specific pages
                        # Create a multiselect widget for page selection
                        selected_pages = st.multiselect(
                            "Select up to 20 specific pages to process:",
                            options=all_page_options,
                            default=[1]  # Default to first page selected
                        )
                        # Check if too many pages are selected
                        if len(selected_pages) > 20:
                            st.warning(f"You've selected {len(selected_pages)} pages. Only the first 20 selected pages will be processed.")
                            selected_pages = selected_pages[:20]
                        # Update the pages list with the specific selections
                        pages_no = selected_pages
                        # Display selected pages
                        if pages_no:
                            st.write(f"Processing {len(pages_no)} specific pages: {pages_no}")
                        else:
                            st.warning("Please select at least one page to process.") 
            else:
                pages_no = [0]  
            myprompt_json = """Please provide a summary first. Double-check the context of all information. Do not discard any information. Extract all information including all values as a structured JSON object with key-value pairs. Include entities, dates, numerical data, relationships, and any other significant information present in the document. Organize related information into nested objects including numbered pages where appropriate for better clarity. Provide translations to English for content in other languages."""
            myprompt_structured = "Please provide a summary on the whole document. Then extract all information including all values and present it in a well-structured format for readability. Include entities, dates, numerical data, relationships, and any other significant information present in the document. Use headings, bullet points, and tables where appropriate. For images, provide a numbered list with page numbers and descriptions. Format the output to be human-readable with clear sections and visual hierarchy. Double-check the context of all information. Do not discard any information. Answer in Markdown."
            myprompt_trans = """Provide a brief summary at the beginning. Then display the complete original content in Markdown format, preserving all tables, font styles, and including placeholders for images/figures. Follow with English and German translations of the full content, maintaining the same formatting structure throughout all versions."""
            myprompt_rechnungde = """Bitte extrahiere aus der angehängten Rechnung folgende Informationen, die für eine Überweisung des Rechnungsbetrags an den Empfänger notwendig sind:
- Ist die Rechnung bereits bezahlt? (key: paid, value: yes/no)
- Rechnungsdatum (key: invoice_date)
- Rechnungsnummer (key: invoice_no)
- Umfang und Art der Lieferung/Leistung (optional, key: subject)
- Steuersatz (optional, key: tax_rate)
- Kundennummer (optional, key: customer_no)
- Rechnungsempfänger (optional, key: invoice_addressee)
- Rechnungsaussteller (key: invoice_issuer)
- Zahlungsempfänger (key: payment_recipient)
- Empfänger IBAN (key: iban)
- Empfänger BIC (key: bic)
- Rechnungswährung (key: currency)
- Betrag in Rechnungswährung (Dezimaltrenner: ",", key: invoice_amount)
- SEPA Verwendungszweck mit Informationen wie Kundennummer und/oder Rechnungsnummer oder Kunden-Referenznummer (Verwendungszweck, key: payment_reference)
- Zahlungsziel (optional, key: due_date)
- Weitere wichtige Informationen zur Rechnung (optional, key: info)
Wichtig: Die Ausgabe aller Informationen einschließlich aller Werte soll als strukturiertes, hierarchisches JSON-Objekt mit Schlüssel-Wert-Paaren erfolgen."""
            myprompt_invoiceen = """Please extract the following information from the attached invoice, which is necessary for transferring the invoice amount to the recipient:
- Is the invoice already paid? (key: paid, value: yes/no)
- Invoice Date (key: invoice_date)
- Invoice Number (key: invoice_no)
- Scope and Type of Delivery/Service (optional, key: subject)
- Tax Rate (optional, key: tax_rate)
- Customer Number (optional, key: customer_no)
- Invoice Addressee (optional, key: invoice_addressee)
- Invoice Issuer (key: invoice_issuer)
- Payment Recipient (key: payment_recipient)
- Recipient IBAN (key: iban)
- Recipient BIC (key: bic)
- Invoice Currency (key: currency)
- Amount in Currency (decimal separator: ".", key: invoice_amount)
- SEPA Purpose of Payment or Payment Refererence like customer number and/or invoice number and/or customer reference number (key: payment_reference)
- Due Date (optional, key: due_date)
- Other important invoice information (optional, key: info)
Important: The output of all information, including all values, should be a structured, hierarchical JSON object with key-value pairs."""
            myprompt_figures = """Please provide a summary on the whole document. For all images, provide a numbered list with the page number and full description. Format the output to be human-readable with clear sections and visual hierarchy. Double-check the context of all information. Do not discard any information. Answer in Markdown."""
            myprompt_tables = """Please provide a page-by-page extraction of all tables, including all table headers, all column names and all cell values. Format requirements: Display the complete original content in HTML format without Markdown elements, preserving all structure, font styles, colors, and include placeholders for images / figures. Always try to maintain the original table structure. Use [?] as placeholders for any unreadable entries. Ensure all tabular information is preserved without omissions. Please verify the accuracy of all extracted information before submitting your response. Do not discard any information. """
            mypromptsel = st.selectbox('Select Task', ('Structured Output', 'JSON Output', 'Explain Figures', 'Extract Tables', 'Translate', 'Invoice (EN)', 'Rechnung (DE)', 'None'))
            if mypromptsel == 'JSON Output':
                myprompt = myprompt_json
                st.session_state.sysprompt_selectbox = 'SchemaSysPrompt'
            elif mypromptsel == 'Translate':
                myprompt = myprompt_trans
                st.session_state.sysprompt_selectbox = 'SchemaSysPrompt'
            elif mypromptsel == 'Structured Output':
                myprompt = myprompt_structured
                st.session_state.sysprompt_selectbox = 'SchemaSysPrompt'
            elif mypromptsel == 'Explain Figures':
                myprompt = myprompt_figures
                st.session_state.sysprompt_selectbox = 'FigureSysPrompt'
            elif mypromptsel == 'Extract Tables':
                myprompt = myprompt_tables
                st.session_state.sysprompt_selectbox = 'SchemaSysPrompt'
            elif mypromptsel == 'Invoice (EN)':
                myprompt = myprompt_invoiceen
                st.session_state.sysprompt_selectbox = 'SchemaSysPrompt'
            elif mypromptsel == 'Rechnung (DE)':
                myprompt = myprompt_rechnungde
                st.session_state.sysprompt_selectbox = 'SchemaSysPrompt'
            else:
                myprompt = ''
            with st.expander('Click here to change IDP parameters'):
                question = st.text_area('Question or Task ', value=myprompt, placeholder='Enter question or task...', key='question_selectbox')
                col1, col2 = st.columns(2)
                with col1:
                    mysysprompt = st.selectbox('Select additional system prompt (may overrule your prompt)', ('SchemaSysPrompt', 'SummarySysPrompt', 'FigureSysPrompt', 'None'), key='sysprompt_selectbox')
                    if mysysprompt == 'SchemaSysPrompt':
                        sysprompt = SystemPrompts().SchemaSysPrompt
                    elif mysysprompt == 'SummarySysPrompt':
                        sysprompt = SystemPrompts().SummarySysPrompt
                    elif mysysprompt == 'FigureSysPrompt':
                        sysprompt = SystemPrompts().FigureSysPrompt
                    else:
                        sysprompt = ''
                with col2:
                    mymodel = st.selectbox('Select LLM', ('Claude Sonnet 3.5 v2', 'Claude Sonnet 3.7', 'AWS NOVA PRO'), key='model_selectbox')
                    if mymodel == 'AWS NOVA PRO':
                        modelId = LanguageModels.NOVA_PRO
                    elif mymodel == 'Claude Sonnet 3.7':
                        modelId = LanguageModels.CLAUDE_SONNET_37
                    else:
                        modelId = LanguageModels.CLAUDE_SONNET_V2

            with st.form("my_form"):
                submitted = st.form_submit_button("Start analysis")
                if submitted is True and tmp_file_path is None:
                    st.error('Error: No file uploaded')
                elif submitted is True and len(question)<1:
                    st.error('Error: Question is empty')
                elif submitted is True and tmp_file_path is not None and len(question)>0:
                    try:
                        rresponse = da_run(tmp_file_path, modelId, pages_no, sysprompt, question)
                        os.unlink(tmp_file_path)
                    except Exception as e:
                        st.error("Error reading file: " + str(e))
                    try:
                        # Parse the response
                        # Assuming response is a dictionary (in "output") with the structure you provided
                        output_data = ast.literal_eval(str(rresponse["output"]))
                        try:
                            if sysprompt == '' and len(output_data) > 1:
                                tab_labels = [f"Page {n+1}" for n in range(len(output_data))]
                                tabs = st.tabs(tab_labels)
                                for n, page in enumerate(output_data):
                                    with tabs[n]:
                                        page = output_data[n]
                                        # Fill each tab with content
                                        st.subheader(f"Page {page['page']}")
                                        # st.write(f"**Detected Languages:** {', '.join(page['detected_languages'])}")
                                        #st.write("**Page Content:**")
                                        #st.write(page['content'], unsafe_allow_html=True)
                                        # Iterate through all keys in the dictionary (except 'page' which we already displayed)
                                        for key, value in page.items():
                                            if key != 'page':  # Skip the page number since we already displayed it
                                                # Format the key as a title (capitalize and replace underscores with spaces)
                                                formatted_key = key.replace('_', ' ').title()
                                                st.write(f"**{formatted_key}:**")
                                                st.markdown(value, unsafe_allow_html=True)
                                                st.write("---")  # Add a separator between sections
                            else:
                                st.write("**Document Content:**")
                                st.markdown(output_data, unsafe_allow_html=True)
                        except Exception as e:
                            st.write("**Document Content:**")
                            st.markdown(output_data, unsafe_allow_html=True)
                    except Exception as e:
                        output_data = str(rresponse["output"])
                        st.markdown(output_data, unsafe_allow_html=True)
                    st.divider()
                    # Display token usage if available
                    try:
                        token_data = ast.literal_eval(str(rresponse["token_usage"]))
                        with st.expander("Token Usage"):
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Input Tokens", token_data["input_tokens"])
                            col2.metric("Output Tokens", token_data["output_tokens"])
                            col3.metric("Total Tokens", token_data["total_tokens"])
                            st.write(f"**Used LLM:** {mymodel}")
                    except Exception as e:
                        st.error("Token usage error: " + str(e))
                    if len(output_data) > 0:
                        st.session_state.output_data = output_data
                    else:
                        st.session_state.output_data = ""
                else:
                    pass
            if len(st.session_state.output_data) > 0:
                with st.container():
                    question = st.text_input(
                        "Chat with the document (Sonnet 3.5 v2)",
                        placeholder="Please enter your question about the document...",
                        value=""
                    )
                    if question:
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
                        # Include context
                        context = ""
                        if st.session_state.get("output_data"):
                            context += f"\n\nDocument Summary: {st.session_state.output_data}"
                            with st.expander("Document Summary (context)"):
                                st.write(context)
                        st.session_state.my_input = f"{SYSTEM_PROMPT_ENGLISH} This is the raw document (only text) <raw>{full_text}</raw> \
                                                      and this is the document summary context: {context}. And here is the question: {question}"
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
