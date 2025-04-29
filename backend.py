from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
from pymongo import MongoClient
from jose import jwt, JWTError
import bcrypt

from inference import tts_generator  # Import TTS model

# ✅ MongoDB Setup (Replace with actual credentials)
MONGO_URI = "mongodb+srv://tts_infernece:dafsdfad232dsf.!@cluster0.6jkfb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["tts_database"]

# ✅ JWT Configuration
SECRET_KEY = "c91f3c3d6f2c4e3a8b1d5e7a9f0c3b6d"  
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# ✅ FastAPI App Initialization
app = FastAPI(title="TTS API")

# ✅ CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Pydantic Models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class TextInput(BaseModel):
    text: str

class FeedbackCreate(BaseModel):
    text: str
    audio_quality: int  # 1-5 rating
    naturalness: int    # 1-5 rating
    comment: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

# ✅ OAuth2 Authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ✅ Token Generation Function
def create_access_token(username: str):
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

# ✅ Get Current User from Token
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.users.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# ✅ User Registration
@app.post("/register")
async def register(user: UserCreate):
    if db.users.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt())
    db.users.insert_one({"username": user.username, "email": user.email, "password": hashed_password})
    
    return {"message": "User registered successfully"}

# ✅ User Login & JWT Token Generation
@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = db.users.find_one({"email": form_data.username})
    if not user or not bcrypt.checkpw(form_data.password.encode(), user["password"]):
        raise HTTPException(status_code=401, detail="Incorrect credentials")

    access_token = create_access_token(user["username"])
    return {"access_token": access_token, "token_type": "bearer"}

# ✅ Text-to-Speech Synthesis (Requires Authentication)
@app.post("/synthesize")
async def synthesize_audio(input: TextInput):
    try:
        audio_data = tts_generator.generate_audio(input.text)
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={'Content-Disposition': 'attachment; filename=synthesized_audio.wav'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Submit Feedback (Requires Authentication)
@app.post("/feedback")
async def create_feedback(feedback: FeedbackCreate, current_user: dict = Depends(get_current_user)):
    feedback_dict = feedback.dict()
    feedback_dict.update({
        "user_id": str(current_user["_id"]),
        "created_at": datetime.utcnow()
    })
    
    db.feedback.insert_one(feedback_dict)
    return {"message": "Feedback submitted"}

# ✅ Retrieve Synthesis History (Requires Authentication)
@app.get("/synthesis-history")
async def get_synthesis_history(current_user: dict = Depends(get_current_user)):
    history = []
    for feedback in db.feedback.find({"user_id": str(current_user["_id"])}):
        history.append({
            "text": feedback["text"],
            "created_at": feedback["created_at"]
        })
    return history

# ✅ Run FastAPI Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
