from PIL import Image
import requests
from transformers import AutoProcessor, BlipForImageTextRetrieval

model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

path = "/home/khoahocmaytinh2022/Desktop/MinhNhut/DATA/caltech-101/101_ObjectCategories/accordion/image_0001.jpg"
image = Image.open(path)
text = "an image of a accordion"

inputs = processor(images=image, text=text, return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits_per_image)  # this is the image-text similarity score