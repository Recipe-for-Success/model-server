from fastapi import FastAPI
from contextlib import asynccontextmanager
from ingredient import IngredientCorpus

corp = IngredientCorpus()

def server_setup(app):
    corp.load_bktree("./corpus/ingredient_bktree.json")
    corp.load_tokens("./corpus/ingredient_tokens.txt")

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
