import base64
import traceback

from PIL import Image, ImageFile
from io import BytesIO
from fastapi import FastAPI, Request, Form, UploadFile
from contextlib import asynccontextmanager
from ingredient import IngredientCorpus
from image_recognition import ImageRecognizer
from urllib import parse

# sometimes the form mangles the images slightly and they don't load without this
ImageFile.LOAD_TRUNCATED_IMAGES = True
# initialize ingredient corpus and image recognizer objects
corp = IngredientCorpus()
image_recognizer = ImageRecognizer()

def server_setup(app):
    """
    Loads the ingredient BK-tree, tokens, and image model
    """
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
    """
    Route for ingredient normalization.

    Accepts string product name ("Vernon's diet cola 64oz pack") and returns a simplified ingredient name ("cola")
    """
    ingr = corp.find_match(raw_product)
    return {"ingredient": ingr}

@app.post("/recognize_image")
async def recognize_image(request: Request):
    """Route to perform image recognition on an ingredient.
    Accepts a base64-encoded image of an ingredient, returns a shortname ingredient label ("tomato")."""
    try:
        # parse JSON data from request body
        data = await request.json()
        image = data.get("data", "")

        # pad input string if needed
        while len(image) % 4 != 0:
            image += "="

        # URL decode the base64 string
        image = parse.unquote(image)
        # then read it to bytes
        base64_bytes = base64.b64decode(image)
        # open bytes as image
        img = Image.open(BytesIO(base64_bytes)).convert("RGB")

        # feed to recognition logic to get label
        label = image_recognizer.recognize_image(img)
        
        # return in expected format
        return {"ingredient": label}

    # in event of an exception, return the error directly
    # quite insecure but convenient for testing
    except Exception as e:
        return {"error": traceback.format_exc()}

