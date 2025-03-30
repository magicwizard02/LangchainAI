import json
import streamlit as st
from pathlib import Path
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.text_splitter import CharacterTextSplitter


st.set_page_config(
    page_title="Quiz GPT",
    page_icon="❓",
)

st.title("QuizGPT")



# function calling 
function = {
    "name": "create_quiz",
    "description": "Generates a quiz with questions and answers.",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"},
                                    "correct": {"type": "boolean"},
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = PromptTemplate.from_template(
    """
    Generate a quiz with 5 multiple-choice questions appropriate for a {difficulty} level (easy, medium, or hard). 
    
    Each question should have 5 answer options labeled A to E, with only one correct answer. 
    
    Present the questions clearly with question numbers and options listed below each question. 
    
    At the end, include an answer key in the format "1: B", "2: D", etc. 
    
    Do not provide explanations—just the questions, options, and the answer key.

    Context: {context}
"""
)


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty):
    chain = prompt | llm
    return chain.invoke({"context": _docs, "difficulty": difficulty})


@st.cache_data(show_spinner="Making Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=2)
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs = None
    topic = None
    openai_api_key = st.text_input("Input your OpenAI API Key")
    source_choice = st.selectbox("Choose Quiz Source:", ["File", "Wikipedia"])
    if source_choice == "File":
        file = st.file_uploader("Upload a .docx, .txt, or .pdf file", type=["pdf", "txt", "docx"])
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)
    difficulty = st.selectbox("Quiz Level", ("EASY", "HRAD"))
if not docs:
   st.markdown(
    """
    Welcome to Quiz GPT
    Ready to challenge your brain?
    Use this chatbot to generate quiz questions, test your knowledge, and check whether your answers are correct.

    Let’s see how much you know — and maybe learn something new along the way!

    Make sure to enter your API Key before using it.
    """
    )

else:
    if not openai_api_key:
        st.error("Please input your OpenAI API Key on the sidebar")
    else:
        llm = ChatOpenAI(
            temperature=0.1,
            streaming=True,
            callbacks=[
                StreamingStdOutCallbackHandler(),
            ],
            openai_api_key=openai_api_key,
        ).bind(
            function_call={
                "name": "create_quiz",
            },
            functions=[
                function,
            ],
        )
    response = run_quiz_chain(docs, topic if topic else file.name, difficulty)
    response = response.additional_kwargs["function_call"]["arguments"]


    # with st.form("questions_form"):
    #     user_answers = {}
    #     questions = json.loads(response)["questions"]
        
    #     for question in questions:
    #         value = st.radio(
    #             "Select an option.",
    #             [answer["answer"] for answer in question["answers"]],
    #             index=None,
    #         )

    #         user_answers[question["question"]] = value  
    #     if {"answer": value, "correct": True} in question["answers"]:
    #                 st.success("Correct!")
    #                 success_count += 1
    #     elif value is not None:
    #                 st.error("Wrong!")

    #     button = st.form_submit_button() 

    # if button:
    #     incorrect_questions = [
    #         question for question in questions
    #         if {"answer": user_answers[question["question"]], "correct": True} not in question["answers"]
    #     ]

    #     if not incorrect_questions:
    #         st.success("You got all answers correct!")
    #         st.balloons()
    #         st.session_state.questions = []  
    #     else:
    #         st.warning("Some answers were incorrect. Retake it.")
    #         st.session_state.questions = incorrect_questions 

    #     button = st.form_submit_button()

    
    with st.form("questions_form"):
        user_answers = {}
        success_count = 0
        questions = json.loads(response)["questions"]

        for i, question in enumerate(questions):
            st.write(f"**{i+1}. {question['question']}**")
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
                key=f"q_{i}"
            )
            user_answers[question["question"]] = value

        button = st.form_submit_button("Submit")

    if button:
        incorrect_questions = []
        for question in questions:
            selected_answer = user_answers.get(question["question"])
            correct_answer = next(
                (a["answer"] for a in question["answers"] if a["correct"]), None
            )

            if selected_answer == correct_answer:
                st.success(f"Correct: {question['question']}")
                success_count += 1
            elif selected_answer is not None:
                st.error(f"Wrong: {question['question']}")
                incorrect_questions.append(question)

        if not incorrect_questions:
            st.success("You got all answers correct!")
            st.balloons()
            st.session_state.questions = []
        else:
            st.warning("Some answers were incorrect. Retake it.")
            st.session_state.questions = incorrect_questions
