import json
from pathlib import Path
from typing import Dict, Any, Optional

INFO_FILE = Path(__file__).resolve().parent / "disease_info.json"

_CACHE: Optional[Dict[str, Any]] = None

def load_disease_database() -> Dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        if INFO_FILE.exists():
            with open(INFO_FILE, "r", encoding="utf-8") as f:
                _CACHE = json.load(f)
        else:
            _CACHE = {}
    return _CACHE

def get_remedy(class_name: str) -> Dict[str, Any]:
    """
    Look up complete management and remedy advisory for a predicted crop disease.
    """
    db = load_disease_database()
    
    # Direct match
    if class_name in db:
        return db[class_name]
    
    # Normalized search (e.g. replacing ___ or underscores)
    normalized_key = class_name.replace(" ", "_")
    for key, val in db.items():
        if key.lower() == normalized_key.lower() or key.lower() == class_name.lower():
            return val
            
    # Generic fallback if custom class is encountered
    parts = class_name.split("___")
    crop = parts[0].replace("_", " ").title() if len(parts) > 1 else "Unknown Crop"
    disease = parts[1].replace("_", " ").title() if len(parts) > 1 else class_name.replace("_", " ").title()
    
    return {
        "crop": crop,
        "disease_name": disease,
        "severity": "Moderate",
        "cause": "Pathogen infection requiring agronomist evaluation.",
        "symptoms": "Leaf tissue discoloration, spotting, or leaf wilting.",
        "favorable_conditions": "Humid, warm or fluctuating environmental conditions.",
        "organic_remedies": [
            "Isolate infected plant sections and prune affected leaves.",
            "Apply broad-spectrum bio-fungicide such as copper soap or neem extract."
        ],
        "chemical_treatments": [
            "Consult with your local agricultural university or extension agent for region-approved fungicides."
        ],
        "prevention": [
            "Maintain sanitized equipment, proper plant spacing, and drip irrigation."
        ]
    }
