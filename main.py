
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if anthropic_api_key and anthropic_api_key.strip() != "":
    llm = ChatAnthropic(api_key=anthropic_api_key, model="claude-3-5-sonnet-20241022")
    print("Using Anthropic Claude model.")
elif openai_api_key and openai_api_key.strip() != "":
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
    print("Using OpenAI GPT model.")
else:
    print("Error: No valid API key found in .env file.")
    exit(1)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful research assistant. Answer the user's query using available tools and provide a concise summary with sources."),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}")
])

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can I help you research? ")
try:
    response = agent_executor.invoke({"query": query})
    print("\n--- Research Result ---\n")
    print(response)
except Exception as e:
    print("Error during agent execution:", e)