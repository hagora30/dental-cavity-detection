
import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

project = rf.workspace("bhoomikashetty").project("dental-cavity-qfnzu")
version = project.version(3)

# Download COCO format into data/raw_coco/
Path("data/raw_coco").mkdir(parents=True, exist_ok=True)
dataset = version.download("coco", location="data/raw_coco")