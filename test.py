import streamlit as st
from PyPDF2 import PdfReader
from PIL import Image     
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator
from gtts import gTTS
from datetime import datetime, timedelta  
from streamlit_autorefresh import st_autorefresh
import pytz
import pytesseract
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="MediAid 💊")

# Sidebar for file upload and language selection
with st.sidebar:
    st.title("🏥 Menu:")
    uploaded_files = st.file_uploader("Upload PDF/Image Files and Click On 'Submit & Process'", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True)
    selected_lang = st.selectbox("👇 Select Language", ["English", "Spanish", "Telugu", "Hindi", "Tamil", "Malayalam"])
    process_files_button = st.button("Submit & Process ✔️")

# Main app title and query input box
st.title(" 🩺 MediAid - Medication Management System 🩹")
st.markdown("#####  Your HealthCare Companion 🧑‍⚕️")
st.markdown("######  Upload prescriptions in PDF or image format to receive detailed analysis, reminders, and language support  🫀.")
user_question = st.text_input(" 👇 Enter your Question ( ex. which medicine should be consumed at night ! )", "")

# Extract text from PDF files
def get_pdf_text(pdf_files):
    """Extract text from multiple PDF files."""
    text = ""
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.warning(f"Could not read one of the PDF files: {e}")
    return text

# Extract text from image files
def get_image_text(image_files):  
    """Extract text from image files using OCR in multiple languages."""
    text = ""
    languages = "eng+spa+tel+hin+tam+mal"
    for image_path in image_files:
        image_data = Image.open(image_path)
        text += pytesseract.image_to_string(image_data, lang=languages)
    return text

# Translate text to the target language
def translate_text(text, target_lang='en'):
    """Translate text to the specified language."""
    return GoogleTranslator(source='auto', target=target_lang).translate(text)

# Sanitize text for audio output
def sanitize_text_for_audio(text):
    """Remove unnecessary symbols and punctuation for clearer audio output."""   
    sanitized_text = re.sub(r'[,\.\*\"\*]', '', text)
    # sanitized_text = re.sub(r'\n', ' ', sanitized_text)
    return sanitized_text

# Generate audio from text
def generate_audio(text, lang='en'):
    """Generate and save audio from text."""
    if not text.strip():
        st.warning("No text available to generate audio.")
        return None
    tts = gTTS(text=text, lang=lang)
    audio_file = "response_audio.mp3"
    tts.save(audio_file)
    return audio_file

# Split long text into manageable chunks
def get_text_chunks(text):
    """Split text into smaller chunks for easier processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

# Initialize session state for reminders if not already done
if "reminders" not in st.session_state:
    st.session_state["reminders"] = {}
    st.session_state["reminder_displayed"] = {"morning": False, "afternoon": False, "evening": False, "night": False}

# Detect medication times from the text
def detect_medication_times(text):
    """Detect specific medication times in the text and return them."""
    times = []
    if "morning" in text.lower():
        times.append("morning")
    if "afternoon" in text.lower():
        times.append("afternoon")
    if "evening" in text.lower():
        times.append("evening")
    if "night" in text.lower():
        times.append("night")
    return times

# Set reminders for specified times
def set_reminders(times):
    """Set reminder times for detected medication times."""
    reminder_intervals = {
        "morning": 9,
        "afternoon": 13,
        "evening": 18,
        "night": 21
    }
    now = datetime.now()
    for time in times:
        reminder_time = now.replace(hour=reminder_intervals[time], minute=0, second=0, microsecond=0)
        if reminder_time < now:
            reminder_time += timedelta(days=1)  # Set for the next day if time has already passed
        st.session_state["reminders"][time] = reminder_time
        st.session_state["reminder_displayed"][time] = False  # Reset reminder display status

# Schedule reminders based on prescription text
def schedule_reminders(text):
    """Detect medication times in text and set reminders."""
    detected_times = detect_medication_times(text)
    if detected_times:
        set_reminders(detected_times)
        st.success("Reminders set based on prescription timing.")  
    else:
        st.warning("No medication times detected in the prescription.")

# Check reminders and trigger if time has been reached
def check_reminders():
    """Check if it's time for any reminders and trigger audio notification."""
    if "reminders" in st.session_state and st.session_state["reminders"]:
        current_time = datetime.now()
        for time_of_day, reminder_time in st.session_state["reminders"].items():
            if current_time >= reminder_time and not st.session_state["reminder_displayed"][time_of_day]:
                reminder_message = f"It's time to take your medication for {time_of_day}."
                st.success(reminder_message)
                for _ in range(3):  # Repeat 3 times
                    st.audio(generate_audio(reminder_message, lang=selected_lang[:2].lower()), format="audio/mp3")
                st.session_state["reminder_displayed"][time_of_day] = True

# Display countdown to the next reminder
def display_next_reminder_countdown():
    """Display countdown timer to the next scheduled reminder."""
    if "reminders" in st.session_state and st.session_state["reminders"]:
        now = datetime.now()
        upcoming_reminders = {k: v for k, v in st.session_state["reminders"].items() if not st.session_state["reminder_displayed"][k]}
        if upcoming_reminders:
            next_reminder_time = min(upcoming_reminders.values())
            countdown = next_reminder_time - now
            hours, remainder = divmod(countdown.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            st.write(f"Next reminder in: {hours}h {minutes}m {seconds}s")

# Load conversational chain for question answering
def get_conversational_chain():
    """Set up the conversational chain with prompt and model."""
    
    prompt_template = """
    You are an expert assistant specializing in analyzing images, reading prescriptions, and processing medical documents in various languages. You can summarize and understand the contents of multiple PDF files and images simultaneously, providing clear and reliable responses. 

    When you receive a query, thoroughly analyze all input files (PDFs or images) to ensure accurate responses. Since these files may contain multiple languages, respond only in the language requested by the user, without mixing other languages unless explicitly specified. 

    Guidelines for your responses:
    1. **Format**: Present information in a clear, structured manner. Use headings, bullet points, and avoid simply copying text. Please provide the entire response in the selected language itself, do not combine multiple languages in the response.
    2. **Accuracy**: Provide detailed information from the files after careful verification. Do not address any safety-related disclaimers; answer confidently.
    3. **Out-of-scope**: If a question does not relate to the context provided in the files, respond with: "Answer is not available in the context."
    4. **Closing statement**: End each response with "You did a Great Job Today, Get Well Soon!!"

    **Context**:
    {context}

    **Question**:
    {question}

    **Answer**:
    
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Process files and handle question
if process_files_button and uploaded_files:
    with st.spinner("Processing..."):
        pdf_files = [f for f in uploaded_files if f.name.endswith('.pdf')]
        image_files = [f for f in uploaded_files if f.name.endswith(('jpg', 'jpeg', 'png'))]

        raw_text = get_pdf_text(pdf_files) + get_image_text(image_files)
        if selected_lang != "English":
            raw_text = translate_text(raw_text, target_lang=selected_lang[:2].lower())

        if raw_text.strip():
            chunks = get_text_chunks(raw_text)
        else:
            st.error("No text could be extracted.")
            chunks = []

        if chunks:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            try:
                vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                st.success("Processing Done!")
            except IndexError:
                st.error("Embedding generation failed. Please check your API setup.")
        else:
            st.warning("No valid text found for embedding. Please upload proper files.")

        # Schedule reminders based on detected times
        schedule_reminders(raw_text)
        st.success("File processing and reminders setup completed.")

    # Answer questions based on extracted information
    if user_question and 'vector_store' in locals():
        chain = get_conversational_chain()
        docs = vector_store.similarity_search(user_question)   
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)["output_text"]

        if selected_lang != "English":
            response = translate_text(response, target_lang=selected_lang[:2].lower())

        st.write(response)
    
        sanitized_response = sanitize_text_for_audio(response)
        audio_file = generate_audio(sanitized_response, lang=selected_lang[:2].lower())
        if audio_file:
            st.audio(audio_file, format="audio/mp3")

    elif not user_question and process_files_button:
        st.warning("Please enter a question to receive a response.")

# Check and display reminders and countdown on each rerun
check_reminders()
display_next_reminder_countdown()

# Auto-refresh the app every 1 minute for periodic checks  # extra added
refresh_script = """             
<script>
    setTimeout(() => {
        window.location.reload();
    }, 60000);  // Refresh every 60 seconds  
</script>
"""
st.markdown(refresh_script, unsafe_allow_html=True)  

st_autorefresh(interval=60 * 1000, key="data_refresh")


# even though i downloaded and integrated the .traindata files of languages english,spanish,telugu,hindi,tamil,malayalam the model is not at all working when i choose the languages spanish and malayalam

# and also the next remainder countdown is only working when i select the language as english but not working for another languages and even the time showcased is not in real time(i.e dynamic)

# and also the model is finding some difficulty when it is given with medicine prescriptions in a language other than english, and providing incorrect answers
