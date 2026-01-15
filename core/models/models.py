import os

import dotenv
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("SI_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("SI_BASE_URL")

agent_model = ChatOpenAI(
    model="MiniMaxAI/MiniMax-M2",
    temperature=0
)

back_agent_model = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3.2",
    temperature=0
)

read_model = ChatOpenAI(
    model="MiniMaxAI/MiniMax-M2",
    temperature=0,
)

grade_model = ChatOpenAI(
    model="MiniMaxAI/MiniMax-M2",
    temperature=0
)


summarize_model = ChatOpenAI(
    model="MiniMaxAI/MiniMax-M2",
    temperature=0
)