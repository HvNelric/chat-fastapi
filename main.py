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
import tempfile
import json
from typing import Dict, Any
#import base64

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
MODEL_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

# Endpoint séparé pour l'audio (utilise les mêmes credentials si non spécifiés)
AZURE_OPENAI_AUDIO_ENDPOINT = os.getenv("AUDIO_ENDPOINT")
AZURE_OPENAI_AUDIO_KEY = os.getenv("AUDIO_API_KEY")
MODEL_AUDIO_DEPLOYMENT_NAME = os.getenv("AUDIO_API_DEPLOYMENT_NAME")

app = FastAPI(debug=True)

# Client principal pour le chat et la vision
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-12-01-preview"
)

# Vérification des credentials audio
if not AZURE_OPENAI_AUDIO_ENDPOINT or not AZURE_OPENAI_AUDIO_KEY:
    raise ValueError("Les credentials audio sont manquants")

# Client séparé pour l'audio
clientAudio = AzureOpenAI(
    api_key=AZURE_OPENAI_AUDIO_KEY,
    azure_endpoint=AZURE_OPENAI_AUDIO_ENDPOINT,
    api_version="2024-12-01-preview"
    #api_version="2024-02-15-preview"
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

class AudioResponse(BaseModel):
    text: str
    response: str

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


#####################################################

async def save_audio_to_temp(audio_data: bytes) -> str:
    """Sauvegarde les données audio dans un fichier temporaire."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
            temp_file.write(audio_data)
            return temp_file.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la sauvegarde du fichier audio: {str(e)}")

async def transcribe_audio(file_path: str) -> str:
    """Transcrit l'audio en utilisant Azure OpenAI Whisper."""
    try:
        with open(file_path, 'rb') as audio_file:
            # Créer un objet file-like avec le nom du fichier
            audio_data = audio_file.read()
            from io import BytesIO
            audio_bytes = BytesIO(audio_data)
            audio_bytes.name = 'audio.webm'  # Important: donner un nom au fichier

            transcript_response = clientAudio.audio.transcriptions.create(
                file=audio_bytes,
                model=MODEL_AUDIO_DEPLOYMENT_NAME,
                language="fr"
            )
            
            print("Transcription réussie")  # Debug
            return transcript_response.text
    except Exception as e:
        print(f"Erreur de transcription détaillée: {str(e)}")  # Log détaillé
        raise HTTPException(status_code=500, detail=f"Erreur de transcription: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Fichier temporaire supprimé: {file_path}")  # Debug

async def get_gpt_response(transcription: str) -> str:
    """Obtient une réponse de GPT-4."""
    try:
        print(f"Envoi à GPT: {transcription}")  # Debug
        chat_response = client.chat.completions.create(  # Utilisation du client principal au lieu de clientAudio
            model=MODEL_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system", 
                    "content": "Vous êtes un assistant utile qui répond en français avec un style décontracté mais professionnel."
                },
                {"role": "user", "content": transcription}
            ],
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        print("Réponse GPT reçue avec succès")  # Debug
        return chat_response.choices[0].message.content
    except Exception as e:
        print(f"Erreur GPT détaillée: {str(e)}")  # Log détaillé
        raise HTTPException(status_code=500, detail=f"Erreur GPT: {str(e)}")

@app.post("/audio", response_model=AudioResponse)
async def process_audio(request: Request):
    """
    Endpoint pour traiter l'audio :
    1. Reçoit le fichier audio
    2. Le transcrit avec Whisper
    3. Envoie la transcription à GPT-4
    4. Retourne la transcription et la réponse
    """
    try:
        # Lire les données audio
        audio_data = await request.body()
        
        # Vérifier si les données audio sont vides
        if not audio_data:
            raise HTTPException(status_code=400, detail="Aucune donnée audio reçue")

        print("Taille des données audio reçues:", len(audio_data))  # Debug

        # Sauvegarder l'audio dans un fichier temporaire
        temp_file_path = await save_audio_to_temp(audio_data)
        print("Fichier temporaire créé:", temp_file_path)  # Debug
        
        # Transcrire l'audio
        transcription = await transcribe_audio(temp_file_path)
        print("Transcription:", transcription)  # Debug
        
        if not transcription:
            raise HTTPException(status_code=422, detail="La transcription a échoué")

        # Obtenir la réponse de GPT
        gpt_response = await get_gpt_response(transcription)
        print("Réponse GPT:", gpt_response)  # Debug

        return AudioResponse(
            text=transcription,
            response=gpt_response
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Erreur inattendue détaillée: {str(e)}")  # Log détaillé
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Endpoint de vérification de l'état du serveur."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "azure_endpoint": AZURE_OPENAI_ENDPOINT is not None,
        "azure_key": AZURE_OPENAI_API_KEY is not None,
        "audio_endpoint": AZURE_OPENAI_AUDIO_ENDPOINT is not None,
        "audio_key": AZURE_OPENAI_AUDIO_KEY is not None,
        "model_deployment": MODEL_DEPLOYMENT_NAME,
        "audio_model_deployment": MODEL_AUDIO_DEPLOYMENT_NAME
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 



#####################################################

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

# completion = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "system", "content": "tu es un assistant efficace"},
#         {"role": "user","content": "ou se trouve annecy ?"},
#     ],
# )

# print(completion.choices[0].message.content)