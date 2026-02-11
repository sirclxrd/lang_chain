from fastapi import FastAPI, Depends, HTTPException, Header
import ollama
import os
from dotenv import load_dotenv


load_dotenv()
API_KEY_CREDITS = {os.getenv("API_KEY"): 10} # i have 10 usage of my api key
# this is simulating the database of all the api keys


app = FastAPI()

def verify_api_key(x_api_key:str = Header(None)):
    credits = API_KEY_CREDITS.get(x_api_key, 0) # if the key doesnt exists return 0 else the value
    if credits <= 0:
        raise HTTPException(status_code=401, detail="Invalid API Key, or no credits")
    return x_api_key

@app.post("/generate")
def generate(prompt: str, x_api_key:str = Depends(verify_api_key)):
    API_KEY_CREDITS[x_api_key] -= 1
    response = ollama.chat(model = "gpt-oss:120b-cloud", messages = [{"role": "user", "content": prompt}])
    return {"response": response["message"]["content"]}

