import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

# wrap into a Python class because the model parameters weren't being properly persisted without doing so
class ImageRecognizer:
    # initialize with pretrained ResNet
    model = models.resnet50(pretrained=True)
    # path to the model parameters

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

    DEVICE = "cpu"

    # these transforms must match the test_transforms used for testing the model
    INPUT_TRANSFORMS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    def reload_model(self, model_path):
        """Function to reload the model parameters from a PyTorch state dictionary.
        Accepts the path to a file created from save"""
        # load pretrained ResNet
        # freeze all but the last layer
        for param in self.model.parameters():
            param.requires_grad = False
        # get number of expected features from ResNet (512)
        num_ftrs = self.model.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
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
        # transform the input image using the model test transforms
        tensor_image = self.INPUT_TRANSFORMS(image).unsqueeze(0)
        # don't update model gradients
        with torch.no_grad():
            # get model output
            self.model.eval()
            output = self.model(tensor_image)
            # get which class is the strongest prediction from the output
            pred = output.data.argmax(dim=1).item()
        # return the name of that class
        return self.CLASS_NAMES[pred]