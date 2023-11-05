import streamlit as st
from streamlit_lottie import st_lottie
from typing import Literal
from dataclasses import dataclass
import json
import base64
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import NLTKTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import nltk
from prompts.prompts import templates
# Audio
from speech_recognition.openai_whisper import save_wav_file, transcribe
from audio_recorder_streamlit import audio_recorder
from aws.synthesize_speech import synthesize_speech
from IPython.display import Audio

def load_lottiefile(filepath: str):

    '''Load lottie animation file'''

    with open(filepath, "r") as f:
        return json.load(f)

st_lottie(load_lottiefile("images/brain.json"), speed=1, reverse=False, loop=True, quality="high", height=300)

#st.markdown("""solutions to potential errors:""")
with st.expander("""Errors when trying to talk to Sensei?"""):
    st.write("""
    This is due to the app's inability to record. Please ensure that your microphone is properly connected, and that you have granted the necessary permission to the browser to access your microphone.""")
with st.expander("""Example Role Profile"""):
    st.write("""
    We are currently looking for suitably experienced and enthusiastic individuals to the position of Consultant - Physician - Medical Oncology at Rockingham General Hospital in Western Australia.

You will lead the multidisciplinary team to provide specialist oncology services to patients. You promote patient safety and quality of care. You will also provide leadership, orientation, training, supervision and education, where relevant, for doctors in training, health service medical practitioners and other health workers.

Key Responsibilities
             
Leads the multidisciplinary team to provide specialist oncology services to patients. Promotes
patient safety and quality of care. Provides leadership, orientation, training, supervision and
education, where relevant, for doctors in training, health service medical practitioners and other
health workers.
             
In collaboration with the Head of Department and other Consultants, works to achieve national,
state and South Metropolitan Health Service (SMHS) performance standards and targets.
Works within the scope of clinical practice as defined and recommended by the SMHS Area
Medical Credentialing Committee (AMCC).

Brief Summary of Duties/Scope of Practice
             
• Each consultant is responsible for the orientation, education and supervision of the junior
medical staff allocated to them. Supervision is especially important during procedures.
             
1. Clinical
1.1 Leads the provision of consumer centred medical care to inpatients and outpatients and
provides a consultation service on request to other patients.
             
1.2 Undertakes clinical shifts at the direction of the Head of Department including participation
in the on-call/after-hours/weekend rosters.
             
1.3 Consults, liaises with and support patients, carers, colleagues, nursing, allied health,
support staff, external agencies and the private sector to provide coordinated
multidisciplinary care.
             
1.4 Responsible for ensuring patients are involved in decision making, regarding their care.
             
1.5 Conducts regular clinical reviews of patients at appropriate intervals with junior doctors
and coordinates patient care with a focus on actively addressing unnecessary delays in
patient admissions, treatment or discharge.
             
1.6 Reviews patients who deteriorate or whose condition is causing concern to hospital staff,
or if requested by the patient or relatives as soon as possible.
             
1.7 Authorises and supports Doctors in Training (DiT’s) in conducting clinical review of all
inpatients daily and to facilitate appropriate early discharges and is generally available for
discussion by phone to assist DiT’s when necessary.
             
1.8 Provides preliminary advice to doctors both internal and external to SMHS and refers
requests for interhospital transfers to the appropriate governance manager advising if
transfer is time critical.
             
1.9 Responsible for the clinical review and clinical management of patients referred to
Outpatient services.
             
1.10 Works with the Head of Department and other Consultants to distribute planned and
unplanned patient demand across the specialty and other hospital sites and champions
clinical service redesign to improve systems of care.
             
1.11 Ensures clinical documentation, including discharge summaries, are completed on time and
undertakes other administrative/management tasks as required.
             
1.12 Participates in departmental and other meetings as required to meet organisational and
service objectives.
             
1.13 Works within the scope of clinical practice as approved by the SMHS Area Medical
Credentialing Committee.
             
1.14 Champions the CanMEDS values and complies with appropriate guidelines for medical
staff.
             
2. Education/Training/Research
             
2.1 Engages in continuing professional development/education and ensures continuous
eligibility for the relevant specialist medical registration.
             
2.2 Educates DiT’s, medical students and other members of the multidisciplinary team through
ward rounds, formal presentations, tutorials and other modalities.
             
2.3 Develops and participates in evidence based clinical research activities relevant to
specialty.
             
2.4 Participates in mandatory training activities to ensure compliance with SMHS policy.
             
2.5 Completes and annual professional development review of their performance with the
Head of Department. """)

st.markdown("""\n""")
jd = st.text_area("""Please enter the job description here, you can find an example one by clicking "Example" at the top of this page.""")
auto_play = st.checkbox("Click this to enable voice responses FROM the AI (Not changeable during the interview)")
#st.toast("4097 tokens is roughly equivalent to around 800 to 1000 words or 3 minutes of speech. Please keep your answer within this limit.")

@dataclass
class Message:
    '''dataclass for keeping track of the messages'''
    origin: Literal["human", "ai"]
    message: str

def autoplay_audio(file_path: str):
    '''Play audio automatically'''
    def update_audio():
        global global_audio_md
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            global_audio_md = f"""
                <audio controls autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
    def update_markdown(audio_md):
        st.markdown(audio_md, unsafe_allow_html=True)
    update_audio()
    update_markdown(global_audio_md)

def embeddings(text: str):

    '''Create embeddings for the job description'''

    nltk.download('punkt')
    text_splitter = NLTKTextSplitter()
    texts = text_splitter.split_text(text)
    # Create emebeddings
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    retriever = docsearch.as_retriever(search_tupe='similarity search')
    return retriever

def initialize_session_state():

    '''Initialize session state variables'''

    if "retriever" not in st.session_state:
        st.session_state.retriever = embeddings(jd)
    if "chain_type_kwargs" not in st.session_state:
        Behavioral_Prompt = PromptTemplate(input_variables=["context", "question"],
                                          template=templates.behavioral_template)
        st.session_state.chain_type_kwargs = {"prompt": Behavioral_Prompt}
    # interview history
    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.history.append(Message("ai", "Hello there! I am your interviewer today. I will access your behavioural fit through a series of questions. Let's get started! Please start by saying hello or introducing yourself. Note: The maximum length of your answer is ~900 words"))
    # token count
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()
    if "guideline" not in st.session_state:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.8, )
        st.session_state.guideline = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type_kwargs=st.session_state.chain_type_kwargs, chain_type='stuff',
            retriever=st.session_state.retriever, memory=st.session_state.memory).run(
            "Create an interview guideline and prepare total of 8 questions. Make sure the questions tests the soft skills")
    # llm chain and memory
    if "conversation" not in st.session_state:
        llm = ChatOpenAI(
        model_name = "gpt-3.5-turbo",
        temperature = 0.8,)
        PROMPT = PromptTemplate(
            input_variables=["history", "input"],
            template="""I want you to act as an interviewer strictly following the guideline in the current conversation.
                            Candidate has no idea what the guideline is.
                            Ask me questions and wait for my answers. Do not write explanations.
                            Ask question like a real person, only one question at a time.
                            Do not ask the same question.
                            Do not repeat the question.
                            Do ask follow-up questions if necessary. 
                            You name is Interview Sensei.
                            I want you to only reply as an interviewer.
                            Do not write all the conversation at once.
                            If there is an error, point it out.

                            Current Conversation:
                            {history}

                            Candidate: {input}
                            AI: """)
        st.session_state.conversation = ConversationChain(prompt=PROMPT, llm=llm,
                                                       memory=st.session_state.memory)
    if "feedback" not in st.session_state:
        llm = ChatOpenAI(
        model_name = "gpt-3.5-turbo",
        temperature = 0.5,)
        st.session_state.feedback = ConversationChain(
            prompt=PromptTemplate(input_variables = ["history", "input"], template = templates.feedback_template),
            llm=llm,
            memory = st.session_state.memory,
        )

def answer_call_back():

    '''callback function for answering user input'''

    with get_openai_callback() as cb:
        # user input
        human_answer = st.session_state.answer
        # transcribe audio
        if voice:
            save_wav_file("temp/audio.wav", human_answer)
            try:
                input = transcribe("temp/audio.wav")
                # save human_answer to history
            except:
                st.session_state.history.append(Message("ai", "Sorry, I didn't get that."))
                return "Please try again."
        else:
            input = human_answer

        st.session_state.history.append(
            Message("human", input)
        )
        # OpenAI answer and save to history
        llm_answer = st.session_state.conversation.run(input)
        # speech synthesis and speak out
        audio_file_path = synthesize_speech(llm_answer)
        # create audio widget with autoplay
        audio_widget = Audio(audio_file_path, autoplay=True)
        # save audio data to history
        st.session_state.history.append(
            Message("ai", llm_answer)
        )
        st.session_state.token_count += cb.total_tokens
        return audio_widget

### ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
if jd:

    initialize_session_state()
    credit_card_placeholder = st.empty()
    col1, col2 = st.columns(2)
    with col1:
        feedback = st.button("Get Interview Feedback")
    with col2:
        guideline = st.button("Interview Guideline")
    audio = None
    chat_placeholder = st.container()
    answer_placeholder = st.container()

    if guideline:
        st.write(st.session_state.guideline)
    if feedback:
        evaluation = st.session_state.feedback.run("please give evalution regarding the interview")
        st.markdown(evaluation)
        st.download_button(label="Download Interview Feedback", data=evaluation, file_name="interview_feedback.txt")
        st.stop()
    else:
        with answer_placeholder:
            voice: bool = st.checkbox("I would like to speak with Interview Sensei")
            if voice:
                answer = audio_recorder(pause_threshold=2.5, sample_rate=44100)
                #st.warning("An UnboundLocalError will occur if the microphone fails to record.")
            else:
                answer = st.chat_input("Your answer")
            if answer:
                st.session_state['answer'] = answer
                audio = answer_call_back()
        with chat_placeholder:
            for answer in st.session_state.history:
                if answer.origin == 'ai':
                    if auto_play and audio:
                        with st.chat_message("assistant"):
                            st.write(answer.message)
                            st.write(audio)
                    else:
                        with st.chat_message("assistant"):
                            st.write(answer.message)
                else:
                    with st.chat_message("user"):
                        st.write(answer.message)

        credit_card_placeholder.caption(f"""
                        Progress: {int(len(st.session_state.history) / 30 * 100)}% completed.
        """)

else:
    st.info("Please submit job description to start interview.")




