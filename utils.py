import json

def load_json(path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)