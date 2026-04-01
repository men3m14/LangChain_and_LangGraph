from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
load_dotenv()





llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools=[TavilySearch()]
agent = create_agent(model=llm, tools=tools)
def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="Find entry level AI jobs on LinkedIn posted in Egypt")})
    print(result)


if __name__ == "__main__":
    main()
