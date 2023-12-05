import base64
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, Request, Form
from contextlib import asynccontextmanager
from ingredient import IngredientCorpus
from image_recognition import ImageRecognizer
from torchvision import models

corp = IngredientCorpus()
image_recognizer = ImageRecognizer()

def server_setup(app):
    # load ingredient BK-tree and tokens
    corp.load_bktree("./corpus/ingredient_bktree.json")
    corp.load_tokens("./corpus/ingredient_tokens.txt")

    # reload model parameters
    image_recognizer.reload_model("./models/ingredient_resnet.pt")

def server_cleanup(app):
    print("Server shutdown")

@asynccontextmanager
async def lifespan(app: FastAPI):
    server_setup(app)
    yield
    server_cleanup(app)

app = FastAPI(lifespan=lifespan)

@app.get("/normalize_ingredient")
async def read_item(raw_product: str):
    ingr = corp.find_match(raw_product)
    return {"ingredient": ingr}

@app.post("/recognize_image")
async def recognize_image(image: str = Form(...)):  # change parameter name to match base64 parameter name in body
    """Function to perform image recognition on an ingredient.
    Accepts a base64-encoded image of an ingredient, returns a shortname ingredient label."""
    # decode base64 to raw bytes
    base64_bytes = base64.b64decode(image)
    # open bytes as image
    img = Image.open(BytesIO(base64_bytes))
    # feed to recognition logic to get label
    label = image_recognizer.recognize_image(img)
    # return in expected format
    return {"ingredient": label}
