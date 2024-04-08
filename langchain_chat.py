#!/usr/bin/env python
# coding: utf-8

# In[15]:


from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import pickle
import os


# In[46]:


class MyCustomOllamaChat:
    def __init__(self, model_name, session_id):
        self.model_name = model_name
        self.session_id = session_id
        self.model = ChatOllama(model=model_name)
        self.store_path = f'./previous_sessions/session_store_{self.model_name.split(":")[0]}.pickle'
        self.__initialize_current_chat()


    def __load_store(self):
        ''' (store = {session_id:chat_content}) load previous chats, or create new if doesn't exist '''
        if not os.path.exists(self.store_path):
            self.store = {}
        else:
            with open(self.store_path, 'rb') as handle:
                self.store = pickle.load(handle)
         

    def __get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        ''' loads a specific previous chat, or creates it if doesn't exist'''
        if self.session_id not in self.store:
            self.store[self.session_id] = ChatMessageHistory()
            with open(self.store_path, 'wb') as handle:
                pickle.dump(self.store, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self.store[self.session_id]


    def __initialize_current_chat(self):
        self.prompt = ChatPromptTemplate.from_messages(
            [
        (
            "system",
            "You're a helpful AI assistant.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{latest_input}"),
            ]
        )

        # parses model output in current chat
        self.parser = StrOutputParser()

        # LCEL chain
        self.runnable = self.prompt | self.model | self.parser

        self.__load_store()
        self.with_message_history = RunnableWithMessageHistory(
        self.runnable,
        self.__get_session_history,
        input_messages_key="latest_input",
        history_messages_key="history",
        )

    def __save_store(self):
        ''' get_response() calls get_session_history() which 
        adds current prompt/response in-memory, 
        this function writes it file'''
        with open(self.store_path, 'wb') as handle:
                pickle.dump(self.store, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def get_response(self, user_input):
        ''' only method (so far) that should be called from outside the class'''
        self.latest_response = self.with_message_history.invoke(
        {"latest_input": user_input},
        config={"configurable": {"session_id": self.session_id}},
        )

        self.__save_store()

        return (self.latest_response)

