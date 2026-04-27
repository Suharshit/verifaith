import torch
from groq import Groq
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv()

print("[VeriFaith] Loading models at startup...")

# Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("[VeriFaith] ✅ Groq client ready")

# Embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print("[VeriFaith] ✅ Embedding model ready")

# NLI model
device = 0 if torch.cuda.is_available() else -1
nli_model = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-base",
    device=device
)
print("[VeriFaith] ✅ NLI model ready")
print("[VeriFaith] All models loaded ✅")