#!/usr/bin/env python
# coding: utf-8

# In[ ]:


''' some initial testing for the project, before using langchain fully, can be ignored '''


# In[ ]:


# this is the linux curl command that is translated to an http request:
# curl http://localhost:11434/api/chat -d '{"model": "gemma:latest", "messages": {'model': 'gemma:latest', 'prompt': 'Why is the sky blue?'}}'


# In[2]:


import requests

# need to run ollama (ollama serve) at this localhost beforehand
url = 'http://localhost:11434/api/generate'  
data = {'model': 'gemma:latest', 'prompt': 'Why is the sky blue?'}  

# the curl -d command corresponds to post (-d to send data as request body)
response = requests.post(url, json=data)  

if response.ok:
    return_data = response.text
else:
    print(f'Request failed with status code: {response.status_code}')
    
    
print (return_data)


# In[ ]:


# langchain can be used to parse the output, instead of trying to parse the returned json stream (apparently can also disable stream, check ollama documentation)


# In[1]:


from langchain.llms import Ollama

ollama = Ollama(base_url='http://localhost:11434', model='gemma:latest')

TEXT_PROMPT = 'Why is the sky blue?'

print(ollama(TEXT_PROMPT))


# In[4]:


# this is also another approach, might use later, but will stick to langchain for now
from llama_index.llms import Ollama

llm = Ollama(model='gemma:latest')
response = llm.complete('Why is the sky blue?')
print(response)

