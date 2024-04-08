#!/usr/bin/env python
# coding: utf-8

# In[ ]:


''' some initial testing for the project, using langchain, can be ignored '''


# In[ ]:


from langchain.llms import Ollama

ollama = Ollama(base_url='http://localhost:11434', model='gemma:latest')

TEXT_PROMPT = 'Why is the sky blue?'

print(ollama(TEXT_PROMPT))


# In[3]:


# LangChain supports many other chat models. Here, we're using Ollama
# note: even though we don't reference the local host url, Ollama server still needs to be running
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# supports many more optional parameters. Hover on your `ChatOllama(...)`
# class to view the latest available supported parameters
llm = ChatOllama(model='gemma:latest')
prompt = ChatPromptTemplate.from_template('Tell me a short joke about {topic}')

# using LangChain Expressive Language chain syntax
# learn more about the LCEL on
# /docs/expression_language/why
chain = prompt | llm | StrOutputParser()

# for brevity, response is printed in terminal
# You can use LangServe to deploy your application for
# production
print(chain.invoke({'topic': 'Space travel'}))


# In[12]:


# can also just do this, construct the chain, and invoke it, without using topic as a placeholder

# supports many more optional parameters. Hover on your `ChatOllama(...)`
# class to view the latest available supported parameters
llm = ChatOllama(model='gemma:latest')
prompt = ChatPromptTemplate.from_template('hello there')

# using LangChain Expressive Language chain syntax
# learn more about the LCEL on
# /docs/expression_language/why
chain = prompt | llm | StrOutputParser()

# for brevity, response is printed in terminal
# You can use LangServe to deploy your application for
# production
print(chain.invoke({}))


# In[34]:


# stream the words as they come in

# supports many more optional parameters. Hover on your `ChatOllama(...)`
# class to view the latest available supported parameters
llm = ChatOllama(model='gemma:latest')
prompt = ChatPromptTemplate.from_template("hi there")

# using LangChain Expressive Language chain syntax
# learn more about the LCEL on
# /docs/expression_language/why
chain = prompt | llm | StrOutputParser()

async for chunk in chain.astream({}):
    print(chunk, end="", flush=True)


# In[3]:


# attempting to add memory so it acts as an actual chat
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

model = ChatOllama(model='gemma:latest')
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

memory = ConversationBufferMemory(return_messages=True)
memory.load_memory_variables({})
{'history': []}

chain = (
    RunnablePassthrough.assign(
        history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
    )
    | prompt
    | model
)

inputs = {"input": "hi im bob"}
response = chain.invoke(inputs)
print (response.content)


# In[53]:


memory.save_context(inputs, {"output": response.content})


# In[54]:


memory.load_memory_variables({})


# In[50]:


inputs = {"input": "what is my name"}
response = chain.invoke(inputs)
print (response.content)


# In[ ]:


# from the documentation, there seems to be a lot of ways to implement memory (aka chat history, to keep the chat going), 
# docs say only CHatMessageHistory ready for production, but even that they seem to implement in various ways, and don't see a way 
# to stream output if you include memory, so will try to implement something custom by mixing and matching from what they have already


# In[1]:


from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

model = ChatOllama(model='gemma:latest')
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a helpful AI assistant.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{latest_input}"),
    ]
)

parser = StrOutputParser()

runnable = prompt | model | parser

# in-memory store (dictionary {session_id:chat_content})
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="latest_input",
    history_messages_key="history",
)


# In[2]:


with_message_history.invoke(
    {"latest_input": "What does cosine mean?"},
    config={"configurable": {"session_id": "123"}},
)


# In[3]:


print (store)


# In[4]:


# asking about previous query
with_message_history.invoke(
    {"latest_input": "summarize the previous answer into one sentence."},
    config={"configurable": {"session_id": "123"}},
)


# In[5]:


print (store)


# In[ ]:


# streaming seems to work somewhat, but there's this extra 'content=' thing, adding a parser in the chain solves this issue
async for chunk in with_message_history.astream({"latest_input": "summarize the previous answer into one sentence."},
    config={"configurable": {"session_id": "123"}}):
    print(chunk, end="", flush=True)


# In[1]:


get_session_history("123")


# In[ ]:


# now I want to parse the chat history, to be able to display it (though at initial run, I can just display the input)
# And build an interface while managing the histories (if overriding existing session_id, give a warning)
# still have the memeory overflow issue (I think look into this first, maybe this can be solved automatically, when web app re-runs python script, this will clear memory)


# In[ ]:





# In[5]:


# trying to deal with memory filling up after few calls
import pynvml

# Initialize NVML
pynvml.nvmlInit()

# Get handle for the first GPU
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Get memory info
info = pynvml.nvmlDeviceGetMemoryInfo(handle)

print(f"Total memory: {info.total / 1024**2} MB")
print(f"Used memory: {info.used / 1024**2} MB")
print(f"Free memory: {info.free / 1024**2} MB")

# Don't forget to shut down NVML
pynvml.nvmlShutdown()

# so no easy way to directly manage this from inside python, but as long as I use the same model should be ok, 
# need to test more (can also check in langchain has solution to this)


# In[ ]:




