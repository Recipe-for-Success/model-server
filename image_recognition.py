import torch
from torch import nn
from torchvision import transforms, models

# wrap into a Python class because the model parameters weren't being properly persisted without doing so
class ImageRecognizer:
    # initialize with pretrained ResNet 50 weights
    model = models.resnet50(pretrained=True)

    # the list of classes to be recognized by the model; update this if you change them
    CLASS_NAMES = [
        "apple",
        "avocado",
        "banana",
        "carrot",
        "cauliflower",
        "cucumber",
        "eggplant",
        "green pepper",
        "lemon",
        "lime",
        "potato",
        "red onion",
        "red pepper",
        "strawberry",
        "tomato",
        "white onion"
    ]
    # use CPU by default because the model runs quickly anyway and EC2 GPU nodes are expensive
    DEVICE = "cpu"

    # input transforms - these must match the test_transforms used for testing the model
    INPUT_TRANSFORMS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    def reload_model(self, model_path):
        """Function to reload the model parameters from a PyTorch state dictionary.
        Accepts the path to the state dictionary to reload into the model."""
        # load pretrained ResNet
        # freeze all but the last layer (which we will replace)
        for param in self.model.parameters():
            param.requires_grad = False
        # get number of expected features from ResNet (512)
        num_ftrs = self.model.fc.in_features
        # create final layer
        # any model architecture changes need to be identical to those used during model training
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(self.CLASS_NAMES)),
        )
        # reload the model state
        # you have to map back to the CPU because the model was trained on a GPU
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def recognize_image(self, image):
        """
        Estimate which of currently 16 ingredients is most likely to be present in an input image.

        Accepts input as an RGB image (*not* RGBA), processes it, and feeds it through the model.

        Returns the class name of the ingredient as a string (e.g. "tomato").
        """
        # transform the input image using the model test transforms, then unsqueeze for non-batch processing
        tensor_image = self.INPUT_TRANSFORMS(image).unsqueeze_(0)
        # don't update model gradients
        with torch.no_grad():
            # get model output (don't train!)
            self.model.eval()
            output = self.model(tensor_image)
            # get which class is the strongest prediction from the output
            pred = output.data.argmax(dim=1).item()
        # return the name of that class
        return self.CLASS_NAMES[pred]
