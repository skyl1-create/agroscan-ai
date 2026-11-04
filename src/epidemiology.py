import requests
import logging
from typing import Dict, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Epidemiological Climate Profiles for Crop Pathogens
# (Optimal Temperature & Relative Humidity ranges for spore germination and incubation)
CLIMATE_PROFILES = {
    "Late_blight": {
        "opt_temp": (12.0, 22.0),
        "min_humidity": 80.0,
        "incubation_days": (3, 5),
        "pathogen_type": "Oomycete (Water Mold)",
        "spread_rate": "Very Rapid (Exponential in high humidity)"
    },
    "Early_blight": {
        "opt_temp": (22.0, 30.0),
        "min_humidity": 65.0,
        "incubation_days": (5, 8),
        "pathogen_type": "Fungus (Alternaria)",
        "spread_rate": "Moderate (Concentric leaf expansion)"
    },
    "Common_rust": {
        "opt_temp": (16.0, 25.0),
        "min_humidity": 75.0,
        "incubation_days": (7, 10),
        "pathogen_type": "Fungus (Puccinia)",
        "spread_rate": "Wind-borne Powdery Spores"
    },
    "Apple_scab": {
        "opt_temp": (13.0, 24.0),
        "min_humidity": 80.0,
        "incubation_days": (9, 14),
        "pathogen_type": "Fungus (Venturia)",
        "spread_rate": "Rain-splash dispersed"
    },
    "Yellow_Leaf_Curl_Virus": {
        "opt_temp": (26.0, 36.0),
        "min_humidity": 30.0,
        "incubation_days": (12, 18),
        "pathogen_type": "Viral (Begomovirus via Whiteflies)",
        "spread_rate": "Vector-mediated systemic infection"
    },
    "Bacterial_spot": {
        "opt_temp": (24.0, 32.0),
        "min_humidity": 75.0,
        "incubation_days": (4, 7),
        "pathogen_type": "Bacterial (Xanthomonas)",
        "spread_rate": "Water-splash & tool contamination"
    }
}

def fetch_live_weather(city_name: str) -> Optional[Dict[str, Any]]:
    """
    Fetches real-time temperature, humidity, and wind speed for any global city
    using free Open-Meteo API (no API key required).
    """
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url, timeout=4).json()
        
        if not geo_res.get("results"):
            return None
            
        first = geo_res["results"][0]
        lat = first["latitude"]
        lon = first["longitude"]
        resolved_name = f"{first.get('name')}, {first.get('country', '')}"
        
        w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
        w_res = requests.get(w_url, timeout=4).json()
        curr = w_res.get("current", {})
        
        return {
            "city": resolved_name,
            "latitude": lat,
            "longitude": lon,
            "temperature": float(curr.get("temperature_2m", 24.0)),
            "humidity": float(curr.get("relative_humidity_2m", 60.0)),
            "precipitation": float(curr.get("precipitation", 0.0)),
            "wind_speed": float(curr.get("wind_speed_10m", 5.0))
        }
    except Exception as e:
        logger.warning("Could not fetch live weather for %s: %s", city_name, e)
        return None

def analyze_climate_fusion(
    predicted_class: str,
    temp_c: float,
    humidity_pct: float,
    damage_pct: float
) -> Dict[str, Any]:
    """
    Multi-Modal Fusion of Visual Diagnosis + Microclimate Humidity/Temp + Foliar Damage Area:
    Calculates epidemiological risk, estimated infection onset time, and critical spray window.
    """
    # Identify matching climate profile
    matched_profile = None
    for key, prof in CLIMATE_PROFILES.items():
        if key.lower() in predicted_class.lower():
            matched_profile = prof
            break
            
    if matched_profile is None:
        # Default fungal profile
        matched_profile = {
            "opt_temp": (18.0, 28.0),
            "min_humidity": 70.0,
            "incubation_days": (5, 9),
            "pathogen_type": "Foliar Pathogen",
            "spread_rate": "Moderate"
        }

    # 1. Environmental Suitability Index (0 to 100%)
    temp_min, temp_max = matched_profile["opt_temp"]
    temp_score = 1.0 if (temp_min <= temp_c <= temp_max) else max(0.2, 1.0 - abs(temp_c - (temp_min + temp_max)/2.0)*0.08)
    
    hum_threshold = matched_profile["min_humidity"]
    if humidity_pct >= hum_threshold:
        hum_score = 1.0
    else:
        hum_score = max(0.2, 1.0 - (hum_threshold - humidity_pct) * 0.025)

    climatic_risk_pct = round((temp_score * 0.45 + hum_score * 0.55) * 100.0, 1)

    # 2. Disease Timeline & Onset Estimation (Host + Climate + Damage)
    inc_min, inc_max = matched_profile["incubation_days"]
    
    # Estimate days since initial infection based on damage progression & humidity
    if damage_pct < 4.0:
        days_infected = f"{inc_min} to {inc_min + 2} days ago"
        biological_stage = "Early Inoculation & Lesion Incubation"
        action_urgency = "Action Window: Next 48–72 Hours (High recovery potential)"
    elif damage_pct < 18.0:
        days_infected = f"{inc_min + 3} to {inc_max + 4} days ago"
        biological_stage = "Active Sporulation & Canopy Expansion"
        action_urgency = "Action Window: Next 24–36 Hours (Critical to stop spore dispersal)"
    else:
        days_infected = f"> {inc_max + 6} days ago"
        biological_stage = "Advanced Epidemic Tissue Necrosis"
        action_urgency = "Emergency Window: Immediate intervention required within 12–24 Hours"

    return {
        "climatic_risk_index": climatic_risk_pct,
        "is_climate_favorable": climatic_risk_pct >= 65.0,
        "pathogen_type": matched_profile["pathogen_type"],
        "spread_rate": matched_profile["spread_rate"],
        "estimated_onset_time": days_infected,
        "biological_stage": biological_stage,
        "action_urgency": action_urgency,
        "climate_diagnosis": (
            f"Current humidity ({humidity_pct}%) and temperature ({temp_c}°C) provide a "
            f"{climatic_risk_pct}% favorable micro-climate for this pathogen's rapid sporulation."
        )
    }

