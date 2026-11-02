import json
import os
from datetime import datetime
from typing import Optional

# Mock database file for inspections
INSPECTION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inspections.json")

def get_equipment_status(equipment_id: str) -> dict:
    """
    Returns mock status for a piece of equipment.
    """
    # Mock logic: Even IDs are OK, Odd IDs have alarms
    try:
        status = "operational" if int(equipment_id.replace("P-", "")) % 2 == 0 else "warning"
    except:
        status = "operational"
    
    return {
        "equipment_id": equipment_id,
        "status": status,
        "last_service_date": "2023-10-15",
        "active_alarms": ["High Temp"] if status == "warning" else []
    }

def log_inspection(equipment_id: str, note: str, severity: str = "low") -> dict:
    """
    Appends an inspection record to a JSON file.
    """
    record = {
        "timestamp": datetime.now().isoformat(),
        "equipment_id": equipment_id,
        "note": note,
        "severity": severity
    }

    # Load existing or create new
    data = []
    if os.path.exists(INSPECTION_FILE):
        try:
            with open(INSPECTION_FILE, "r") as f:
                data = json.load(f)
        except:
            data = []
    
    data.append(record)
    
    with open(INSPECTION_FILE, "w") as f:
        json.dump(data, f, indent=2)
        
    return {"success": True, "message": f"Inspection logged for {equipment_id}"}