import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "src", "data", "raw")
FULL_DATA_PATH = DATA_PATH

print(os.listdir())