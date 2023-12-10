# Model Server

## Installing and running

This repository contains the FastAPI machine learning server that hosts the ingredient image recognition model and the Python NLP logic for simplifying product names into ingredients.

This server needs to be available on localhost port 8000 on the same machine as the Pic to Plate NextJS server.

Running the server should not require anything more than installing the requirements packages

> pip install -r requirements.txt

and then launching the model server
> uvicorn server:app --reload

## Updating the machine learning model
On launch, the server automatically reloads the image model from the PyTorch state dictionary in `models/ingredient_resnet.pt`.

As long as you do not change the ingredient classes or the use of normalization you can simply replace this with a new state dict from the training notebooks and restart the server.

If you change the **ingredient classes to be recognized**, you also need to change the class name array on line 12 of `image_recognition.py` to reflect the new classes.

If you change the **data augmentations**, you need to adjust the INPUT_TRANSFORMS on line 32 of `image_recognition.py` to match the test transforms used during training.

Finally, if you change the **model architecture**, you need to edit the model construction logic in `reload_model()` on line 39 of `image_recognition.py` to match the new logic or you will get an error when you try to reload the state dict.