import json
from datetime import datetime
from pathlib import Path


def save_training_log(log_data, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    log_data['timestamp'] = datetime.now().isoformat()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)


def load_training_log(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
