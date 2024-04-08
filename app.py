#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify, render_template
from langchain_chat import MyCustomOllamaChat


# In[4]:


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    
    print (data)
    
    session_id = data['sessionId']
    model_name = data['chosenModel']
    latest_prompt = data['message']
    this_chat = MyCustomOllamaChat(model_name, session_id)
    response = this_chat.get_response(latest_prompt)

    print (response)

    return jsonify(response=response) 


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


# In[ ]:




