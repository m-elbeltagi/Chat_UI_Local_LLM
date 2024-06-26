# Chat UI for LLMs Running Locally

It is becoming increasingly feasible to run LLMs locally, especially using tools that facilitate downloading and managing different versions of open source LLMs like Ollama. However, Ollama only allows interaction with a local model via the command line, and with no access to previous chat sessions. Here I attempt to make this interaction easier by creating a chat interface for locally running LLMs, which also saves previous chats, and allows resuming them.

I started by trying to send requests to the Ollama API directly, and parsing the response. I soon found that the LangChain framework has many built-in functionalities to facilitate this, so I switched to interacting with the Ollama API via LangChain, and constructing a "chat with memory" pipeline. Using Flask I turned this into a web application, with a simple HTML that displayes the interface, sends & receives data (using AJAX) from the Flask app, and displayes the model's output. 

![CHAT_UI](static/images/Chat_UI.png) 

## How To Use

1) First install Ollama. On Linux run:

``` curl -fsSL https://ollama.com/install.sh | sh ``` ,      or download the Windows installer: https://ollama.com/download/windows
  

2) Install the model of interest:

``` ollama pull gemma:latest  ``` ,        or whatever other model you want (Gemma has very good performance to size ratio, making it ideal for running locally).


3) Install the packages in the requirements.txt file (probably best to use a dedicated environment for this, to avoid conflicts):

``` pip install -r requirements.txt ```
  

4) Start the Ollama server:

``` Ollama serve ```


5) Run app.py, then open the local host url

```app.py```


Steps **_4_** and **_5_** need to be done everytime to start the app.

The **_Session ID_** is unique, and automatically saves the current chat under that **_Session ID_** (or creates a new **_Session ID_** with the current chat if it didn't already exist). If the same **_Session ID_** is later entered, it automatically resumes that chat, so the model will have access to all the previous exchanges that were made under that **_Session ID_**.

# 
For more info on Ollama, and how to manage installed models, e.g., customize the system prompt for installed models, manage GPU vs CPU usage, etc., see: https://github.com/ollama/ollama
#



## To-Do List
This is just a prototype, the next few functionalitities I want to add are:
- erase old chats
- view old chats (should auto load if **_Session ID_** already exists)
- parse old chats, so viewing isn't painful (langchain built-in methods for this?)
- stream model responses (this already works in initial_testing_2.py file, but output needs some parsing)
- add way to automatically include new models to options
- "prettify" the web page



