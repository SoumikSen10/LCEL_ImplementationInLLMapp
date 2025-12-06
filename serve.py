from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes

import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model = "llama-3.1-8b-instant", groq_api_key = groq_api_key)

# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])

parser = StrOutputParser()

# Create chain
chain = prompt_template | model | parser


# App definition
app = FastAPI(title = "Langchain Server",
              version = "1.0",
              description = "A simple API server using Langchain runnable interfaces")

# Adding chain routes
add_routes(app, 
           chain, 
           path = "/chain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "localhost", port = 8000)


# hit url with : http://localhost:8000/docs to access the chain endpoint



'''
Chaining in LangChain means connecting multiple LLM steps—like the prompt, the model, and the output parser—into a single pipeline using the `|` operator. Instead of manually formatting prompts, calling the model, and parsing the response every time, a chain bundles these steps into one reusable function that can be invoked easily. This makes the code cleaner, avoids repetition, and allows LangServe or FastAPI to expose the entire workflow as an API endpoint. Essentially, chaining turns several LLM operations into one smooth, modular, and efficient process.
'''