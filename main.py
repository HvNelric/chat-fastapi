from dotenv import load_dotenv
import os

from fastapi import HTTPException
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AzureOpenAI
from typing import List
import asyncio
import base64

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
MODEL_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

app = FastAPI(debug=True)

# Initialisation du client Azure OpenAI
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-12-01-preview"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # URL de votre frontend Angular
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle Pydantic pour les messages
class Message(BaseModel):
    role: str  # "user" ou "assistant"
    content: str
    #image: Optional[str] = None 

class ChatRequest(BaseModel):
    messages: list[dict[str, str]]


async def generate_stream(messages: list[dict[str, str]]):
    try:
        response = client.chat.completions.create(
            model=MODEL_DEPLOYMENT_NAME,  # ou votre modèle préféré
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=2000
        )
  
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                await asyncio.sleep(0.05)
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"Erreur: {str(e)}"

@app.post("/chat")
async def chat_stream(request: ChatRequest):
    try:
        return StreamingResponse(
            generate_stream(request.messages),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

#####################################################

async def img_response(messages: list[dict[str, str]]):
    
    try: 
        for msg in messages:
            if msg["role"] == "user":
                if "image" in msg:
                    response = client.chat.completions.create(
                        model=MODEL_DEPLOYMENT_NAME,  # ou votre modèle préféré
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": msg['content']},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": msg["image"]  
                                        }
                                    }
                                ]
                            }
                        ],
                        stream=True,
                        temperature=0.7,
                        max_tokens=2000
                    )   
            else:
                response = client.chat.completions.create(
                    model=MODEL_DEPLOYMENT_NAME,  # ou votre modèle préféré
                    messages=messages,
                    stream=True,
                    temperature=0.7,
                    max_tokens=2000
                )    
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                await asyncio.sleep(0.05)
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"Erreur: {str(e)}"


@app.post("/ocr")
async def img_stream(request: ChatRequest):
    try:
        return StreamingResponse(
            img_response(request.messages),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# completion = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "system", "content": "tu es un assistant efficace"},
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "que vois-tu sur l'image ?"},
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": img_url
#                     },
#                 },
#             ],
#         },
#     ],
# )

# print(completion.choices[0].message.content)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "tu es un assistant efficace"},
        {"role": "user","content": "ou se trouve annecy ?"},
    ],
)

print(completion.choices[0].message.content)