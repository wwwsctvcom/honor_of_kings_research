import io
import base64
import torch
from PIL import Image
from loguru import logger
from pathlib import Path
from model.model_config import ResNetModelConfig
from model.model_resnet import AutoGameForImageClassification
from fastapi import FastAPI
from singleton_decorator import singleton
from utils.datasets import ImagePreprocess
from utils.tools import get_available_device
from pydantic import BaseModel


@singleton
class Model:
    """
    cache model
    """

    def __init__(self, model_name_or_path: str = None, requires_grad: bool = False):
        resnet_config = ResNetModelConfig()
        self.model = AutoGameForImageClassification(config=resnet_config)

        if Path(model_name_or_path).exists():
            logger.info(f"Loading main model from: {model_name_or_path}")
            self.model.load_state_dict(torch.load(model_name_or_path, map_location=get_available_device()))

        # set weights trainable
        self.model.requires_grad_(requires_grad)

    def predict(self, image):
        transform = ImagePreprocess()
        image = transform(image).unsqueeze(0)

        action_logits, _ = self.model(image)
        action = torch.argmax(torch.softmax(action_logits, dim=-1), dim=-1).squeeze().item()
        return action


# load model
MODEL_PATH = "./weights/main_best.pth"
model = Model(model_name_or_path=MODEL_PATH)
app = FastAPI()


class ImageInput(BaseModel):
    image: str


class JSONResponse(BaseModel):
    label: int
    status_code: int = 200
    content: str | None = None


@app.post("/classify")
async def classify(image_input: ImageInput):
    try:
        decoded_image = base64.b64decode(image_input.image)
        image = Image.open(io.BytesIO(decoded_image))

        result = model.predict(image)
        return JSONResponse(label=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
