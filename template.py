import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "cnnClassifier"

list_of_files = [
    # ".github/workflows/.gitkeep",
    # f"src/{project_name}/__init__.py",
    # f"src/{project_name}/components/__init__.py",
    # f"src/{project_name}/utils/__init__.py",
    # f"src/{project_name}/utils/common.py",
    # f"src/{project_name}/config/__init__.py",
    # f"src/{project_name}/config/configuration.py",
    # f"src/{project_name}/pipeline/__init__.py",
    # f"src/{project_name}/entity/__init__.py",
    # f"src/{project_name}/entity/config_entity.py",
    # f"src/{project_name}/constants/__init__.py",
    # "dvc.yaml",
    # "params.yaml",
    # "setup.py",
    # "templates/index.html",
    # "utils.py",
    
    "config/config.py",
    "requirements.txt",
    "notebooks/speech_emotion_recognition.ipynb",
    'scripts/.gitkeep',
    "pipelines/pipeline.py",
    "main.py",
    "app.py",
    "models/.gitkeep",
    "artifacts/dataset/.gitkeep"
]


for filepath in list_of_files:
    filepath = Path(filepath) # to solve the windows path issue
    filedir, filename = os.path.split(filepath) # to handle the project_name folder


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")