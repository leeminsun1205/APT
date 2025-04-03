from PIL import Image
import requests
from transformers import AutoProcessor, BlipForImageTextRetrieval

model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

url = "./DATA/caltech-101/101_ObjectCategories/accordion/image_0001.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "an image of a accordion"

inputs = processor(images=image, text=text, return_tensors="pt")
outputs = model(**inputs)