MAP = {
    "AMBULANCE": "Ambulance",
    "AMBULANCE SERVICE": "Ambulance",
    "AMBULATORY SURGICAL CENTER": "Ambulatory Surgical Center",
    "AMBULATORY TRANSPORTATION SERVICES": "Ambulatory Transportation",
    "ANATOM": "Anatomy / Pathology",
    "ADVANCED RN PRACT": "Advanced RN Practice",
    "ADVANCED REGISTERED NURSE PRAC": "Advanced RN Practice",
}


def norm_specialty(x):
    x = (x or "").strip()
    if not x:
        return "Unknown"
    up = x.upper()
    return MAP.get(up, x.title())
