import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENV_FPATH = os.path.join(ROOT_DIR, ".env")

CODE_DIR = os.path.join(ROOT_DIR, "src")

OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

DATABASE_DIR = os.path.join(ROOT_DIR, "db")

DATA_DIR = os.path.join(ROOT_DIR, "data")
