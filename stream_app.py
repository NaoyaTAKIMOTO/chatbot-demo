from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, load_tools, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

import streamlit as st
import os
import json


search = GoogleSearchAPIWrapper()
wolfram = WolframAlphaAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Math",
        func=wolfram.run,
        description="useful for when you need to solve mathematical question"
    )
]
prefix = """Have a conversation with a human, answering the following questions as best you can.
      You think in Japanese. 
      You have access to the following tools:"""
suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"]
)
memory = ConversationBufferMemory(memory_key="chat_history")


def generate_text_with_memory(question):

    llm_chain = LLMChain(llm=OpenAI(temperature=0.5), prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True,
                                                     memory=memory,
                                                     max_iterations=4, early_stopping_method="generate")
    result = agent_chain.run(input=question)
    return result


st.session_state.tweet = ""
st.session_state.text_error = ""

st.set_page_config(page_title="Tweet", page_icon="🤖")
if "tweet" not in st.session_state:
    st.session_state.tweet = ""
if "image" not in st.session_state:
    st.session_state.image = ""
if "text_error" not in st.session_state:
    st.session_state.text_error = ""
if "image_error" not in st.session_state:
    st.session_state.image_error = ""
if "feeling_lucky" not in st.session_state:
    st.session_state.feeling_lucky = False
if "n_requests" not in st.session_state:
    st.session_state.n_requests = 0

# Force responsive layout for columns also on mobile
st.write(
    """<style>
    [data-testid="column"] {
        width: calc(50% - 1rem);
        flex: 1 1 calc(50% - 1rem);
        min-width: calc(50% - 1rem);
    }
    </style>""",
    unsafe_allow_html=True,
)

# Render Streamlit page
st.title("生成型AIデモ")
st.markdown(
    """
    ## GPT-3 のデモ
    プロンプトによる反応の違い
    """
)

question = st.text_input(
    label="質問文", placeholder="シャア専用ザクより遅いモビルスーツを列挙してください。")

answer = ""
col1, col2 = st.columns(2)

if st.button(
    label="質問する",
    type="primary",
):
    answer = generate_text_with_memory(question)
st.write('Answer:', answer)
if len(answer) > 0:
    answer2 = ""
    question2 = st.text_input(label="追加で質問文", placeholder="日本語で答えてください")
    if st.button(
        label="質問をする",
        type="primary",
        key="question2"
    ):
        answer2 = generate_text_with_memory(question2)
    st.write('Answer2:', answer2)
