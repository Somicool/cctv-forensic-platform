"""Demo Vehicle Registry - OFFLINE, SYNTHETIC. NOT a real police database.

Generates ONE permanent, realistic-looking Indian RC-style record per unique
recognised number plate. Records are:
  * deterministic (same plate -> same data, seeded from the plate),
  * created once and stored permanently (SQLite table + JSON mirror),
  * never regenerated, and changed ONLY by a manual edit.

Provider abstraction
--------------------
`get_provider()` returns a RegistryProvider. The demo implementation synthesises
data; a real police-database API can be added later as another provider with the
SAME method surface + response shape, so the routes and the frontend never change
(set config.REGISTRY_PROVIDER = "police_api" and register it here).

This module does NOT touch OCR, tracking, or search - it only stores/serves
registry records keyed by plate.
"""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta, timezone

from . import config, database

# ----------------------------------------------------------------- vocab
_MALE = ["Amit", "Rajesh", "Suresh", "Vijay", "Ramesh", "Sanjay", "Anil", "Deepak",
         "Manoj", "Prakash", "Rahul", "Vikram", "Arjun", "Kiran", "Nitin", "Ashok",
         "Sunil", "Harish", "Mahesh", "Jayesh", "Bharat", "Chirag", "Mehul", "Paresh"]
_FEMALE = ["Priya", "Anita", "Sunita", "Kavita", "Neha", "Pooja", "Meena", "Rekha",
           "Sneha", "Divya", "Nisha", "Rita", "Geeta", "Jyoti", "Komal", "Manisha",
           "Payal", "Sheetal", "Krupa", "Hetal", "Bhavna", "Falguni"]
_SURNAMES = ["Patel", "Sharma", "Shah", "Desai", "Mehta", "Joshi", "Gupta", "Verma",
             "Singh", "Reddy", "Nair", "Rao", "Kumar", "Chauhan", "Trivedi", "Bhatt",
             "Modi", "Solanki", "Parmar", "Vyas", "Pandya", "Thakkar"]

# plate state-code -> (state, [cities], example RTO city by first RTO digit-ish)
_STATES = {
    "GJ": ("Gujarat", ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"]),
    "MH": ("Maharashtra", ["Mumbai", "Pune", "Nagpur", "Nashik", "Thane"]),
    "DL": ("Delhi", ["New Delhi", "Rohini", "Dwarka", "Janakpuri"]),
    "KA": ("Karnataka", ["Bengaluru", "Mysuru", "Mangaluru", "Hubli"]),
    "RJ": ("Rajasthan", ["Jaipur", "Jodhpur", "Udaipur", "Kota"]),
    "UP": ("Uttar Pradesh", ["Lucknow", "Kanpur", "Noida", "Agra", "Varanasi"]),
    "TN": ("Tamil Nadu", ["Chennai", "Coimbatore", "Madurai", "Salem"]),
    "MP": ("Madhya Pradesh", ["Bhopal", "Indore", "Gwalior", "Jabalpur"]),
    "WB": ("West Bengal", ["Kolkata", "Howrah", "Siliguri", "Durgapur"]),
    "HR": ("Haryana", ["Gurugram", "Faridabad", "Panipat", "Ambala"]),
    "PB": ("Punjab", ["Ludhiana", "Amritsar", "Jalandhar", "Patiala"]),
    "TS": ("Telangana", ["Hyderabad", "Warangal", "Karimnagar"]),
    "AP": ("Andhra Pradesh", ["Visakhapatnam", "Vijayawada", "Guntur"]),
    "KL": ("Kerala", ["Thiruvananthapuram", "Kochi", "Kozhikode"]),
    "RJ2": ("Rajasthan", ["Jaipur"]),
}
_AREAS = ["Nr. Railway Station", "Opp. City Mall", "Sector 12", "Ring Road",
          "Station Road", "Gandhi Nagar", "Nr. Bus Depot", "Old Market Area",
          "Shivaji Chowk", "Nehru Colony", "Industrial Estate", "Green Park"]

# vehicle_type -> (vehicle_class, [brands], {brand:[models]}, [fuels])
_VEHICLES = {
    "Motorcycle": ("MCWG (Motorcycle With Gear)",
                   ["Hero", "Honda", "Bajaj", "TVS", "Royal Enfield", "Yamaha"],
                   {"Hero": ["Splendor Plus", "HF Deluxe", "Passion Pro"],
                    "Honda": ["Shine", "Unicorn", "SP 125"],
                    "Bajaj": ["Pulsar 150", "Platina", "CT 100"],
                    "TVS": ["Apache RTR", "Star City", "Raider"],
                    "Royal Enfield": ["Classic 350", "Bullet 350", "Hunter 350"],
                    "Yamaha": ["FZ", "MT-15", "R15"]},
                   ["Petrol"]),
    "Scooter": ("MCWOG (Scooter)",
                ["Honda", "TVS", "Suzuki", "Hero", "Bajaj"],
                {"Honda": ["Activa 6G", "Dio", "Activa 125"],
                 "TVS": ["Jupiter", "NTorq", "Scooty Pep+"],
                 "Suzuki": ["Access 125", "Burgman"],
                 "Hero": ["Pleasure+", "Maestro Edge"],
                 "Bajaj": ["Chetak (EV)"]},
                ["Petrol", "Electric"]),
    "Auto-Rickshaw": ("Three Wheeler (Passenger)",
                      ["Bajaj", "Piaggio", "Mahindra", "TVS"],
                      {"Bajaj": ["RE Compact", "Maxima Z"],
                       "Piaggio": ["Ape City", "Ape Xtra"],
                       "Mahindra": ["Alfa Passenger", "Treo (EV)"],
                       "TVS": ["King Deluxe"]},
                      ["CNG", "Petrol", "Electric"]),
    "Car": ("LMV (Light Motor Vehicle)",
            ["Maruti Suzuki", "Hyundai", "Tata", "Mahindra", "Honda", "Toyota", "Kia"],
            {"Maruti Suzuki": ["Swift", "Wagon R", "Baleno", "Dzire", "Ertiga"],
             "Hyundai": ["i20", "Creta", "Venue", "Grand i10"],
             "Tata": ["Nexon", "Punch", "Tiago", "Altroz"],
             "Mahindra": ["Scorpio", "XUV700", "Bolero", "Thar"],
             "Honda": ["City", "Amaze"],
             "Toyota": ["Innova Crysta", "Fortuner", "Glanza"],
             "Kia": ["Seltos", "Sonet"]},
            ["Petrol", "Diesel", "CNG"]),
    "Tempo": ("LGV (Light Goods Vehicle)",
              ["Tata", "Mahindra", "Force"],
              {"Tata": ["Ace Gold", "Intra V30"], "Mahindra": ["Jeeto", "Bolero Maxi"],
               "Force": ["Traveller", "Kargo King"]}, ["Diesel", "CNG"]),
    "Mini-Truck": ("LGV (Light Goods Vehicle)",
                   ["Tata", "Mahindra", "Ashok Leyland"],
                   {"Tata": ["Ace HT", "Yodha"], "Mahindra": ["Bolero Pik-Up", "Jeeto"],
                    "Ashok Leyland": ["Dost+", "Bada Dost"]}, ["Diesel", "CNG"]),
    "Pickup": ("LGV (Light Goods Vehicle)",
               ["Mahindra", "Tata", "Isuzu"],
               {"Mahindra": ["Bolero Pik-Up", "Supro"], "Tata": ["Yodha", "Intra"],
                "Isuzu": ["D-Max"]}, ["Diesel"]),
    "Truck": ("HGV (Heavy Goods Vehicle)",
              ["Tata", "Ashok Leyland", "Eicher", "BharatBenz"],
              {"Tata": ["LPT 1613", "Signa 2823"], "Ashok Leyland": ["Ecomet", "Boss"],
               "Eicher": ["Pro 2049", "Pro 3015"], "BharatBenz": ["1917R"]}, ["Diesel"]),
    "Bus": ("HPMV (Heavy Passenger Motor Vehicle)",
            ["Tata", "Ashok Leyland", "Volvo", "Eicher"],
            {"Tata": ["Starbus", "LP 909"], "Ashok Leyland": ["Viking", "Lynx"],
             "Volvo": ["9400"], "Eicher": ["Skyline Pro"]}, ["Diesel", "CNG"]),
}
_COLORS = ["White", "Silver", "Grey", "Black", "Blue", "Red", "Maroon",
           "Brown", "Green", "Yellow", "Orange"]
_VIOLATIONS = ["Over-speeding", "Signal jump", "No helmet", "Triple riding",
               "Wrong-side driving", "No parking zone", "Expired PUC",
               "Using phone while driving"]

# map a detected class label -> a registry vehicle_type (hint at ingestion)
_HINT_MAP = {
    "motorcycle": "Motorcycle", "scooter": "Scooter", "auto-rickshaw": "Auto-Rickshaw",
    "car": "Car", "truck": "Truck", "bus": "Bus", "tempo": "Tempo",
    "mini-truck": "Mini-Truck", "pickup": "Pickup", "lcv": "Tempo", "hcv": "Truck",
    "bicycle": "Motorcycle",
}


def _norm(plate: str) -> str:
    return "".join(ch for ch in (plate or "").upper() if ch.isalnum())


def _pretty_plate(norm: str) -> str:
    """GJ23EB1224 -> 'GJ 23 EB 1224' when it fits the Indian pattern."""
    import re
    m = re.match(r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$", norm)
    return " ".join(m.groups()) if m else norm


def _mask(prefix: str, tail_digits: str) -> str:
    return f"{prefix}{'*' * 8}{tail_digits}"


def _mobile(rng) -> str:
    return "+91 " + str(rng.choice("6789")) + "".join(str(rng.randint(0, 9)) for _ in range(9))


def generate_record(plate: str, hint: str | None = None) -> dict:
    """Deterministic synthetic RC-style record for a plate (seeded by the plate)."""
    norm = _norm(plate)
    seed = int(hashlib.sha256(norm.encode()).hexdigest()[:12], 16)
    rng = random.Random(seed)

    import re
    m = re.match(r"^([A-Z]{2})(\d{1,2})", norm)
    code2 = m.group(1) if m else rng.choice(list(_STATES))
    rto2 = m.group(2) if m else f"{rng.randint(1, 40):02d}"
    state, cities = _STATES.get(code2, ("Gujarat", ["Ahmedabad", "Surat", "Vadodara"]))
    city = rng.choice(cities)

    vtype = _HINT_MAP.get((hint or "").lower()) or rng.choice(list(_VEHICLES))
    vclass, brands, models, fuels = _VEHICLES[vtype]
    brand = rng.choice(brands)
    model = rng.choice(models[brand])
    fuel = rng.choice(fuels)
    color = rng.choice(_COLORS)

    gender = rng.choice(["Male", "Male", "Female"])          # slight male skew (realistic RC)
    first = rng.choice(_MALE if gender == "Male" else _FEMALE)
    surname = rng.choice(_SURNAMES)
    father = rng.choice(_MALE) + " " + surname

    reg_dt = datetime.now(timezone.utc) - timedelta(days=rng.randint(120, 4700))
    reg_year = reg_dt.year
    ins_ok = rng.random() > 0.18
    puc_ok = rng.random() > 0.22
    ins_dt = (reg_dt + timedelta(days=rng.randint(300, 3000)))
    puc_dt = (datetime.now(timezone.utc) + timedelta(days=rng.randint(-90, 300)))

    stolen = rng.random() < 0.05
    blacklisted = rng.random() < 0.06
    n_viol = rng.choice([0, 0, 0, 1, 1, 2, 3])
    violations = rng.sample(_VIOLATIONS, n_viol) if n_viol else []

    dl_no = f"{code2}{rto2}{reg_year - rng.randint(0, 8)}{rng.randint(0, 9999999):07d}"
    chassis = _mask("MA" + rng.choice("13579") + rng.choice("ABCDEFGH"),
                    f"{rng.randint(0, 9999):04d}")
    engine = _mask(rng.choice("JKLMN") + rng.choice("ABCDE"),
                   f"{rng.randint(0, 9999):04d}")
    emp_name = rng.choice(_MALE + _FEMALE) + " " + surname

    return {
        "plate_normalized": norm,
        "vehicle_number": _pretty_plate(norm),
        "owner_name": f"{first} {surname}",
        "father_name": father,
        "gender": gender,
        "mobile_number": _mobile(rng),
        "alternate_mobile": _mobile(rng),
        "driving_license_no": dl_no,
        "vehicle_type": vtype,
        "vehicle_brand": brand,
        "vehicle_model": model,
        "vehicle_color": color,
        "fuel_type": fuel,
        "registration_date": reg_dt.strftime("%d-%m-%Y"),
        "registration_state": state,
        "registration_office": f"RTO {city} ({code2}-{rto2})",
        "address": f"{rng.randint(1, 499)}, {rng.choice(_AREAS)}",
        "city": city,
        "district": city,
        "state": state,
        "pin_code": f"{rng.randint(11, 79)}{rng.randint(0, 9999):04d}",
        "insurance_status": ("Valid (till %s)" % ins_dt.strftime("%d-%m-%Y")) if ins_ok
                            else ("Expired (%s)" % ins_dt.strftime("%d-%m-%Y")),
        "puc_status": ("Valid (till %s)" % puc_dt.strftime("%d-%m-%Y")) if puc_ok
                     else "Expired",
        "vehicle_class": vclass,
        "chassis_number": chassis,
        "engine_number": engine,
        "rc_status": "Active" if rng.random() > 0.05 else "Suspended",
        "blacklist_status": "Blacklisted (DEMO)" if blacklisted else "No",
        "stolen_status": "Reported Stolen (DEMO)" if stolen else "No",
        "previous_violations": violations,
        "previous_investigation_count": rng.choice([0, 0, 0, 1, 1, 2]),
        "emergency_contact": f"{emp_name} · {_mobile(rng)}",
        "notes": "Synthetic DEMO record - not real RTO/police data. For demonstration only.",
        "is_demo": True,
        "source": "demo",
    }


# ----------------------------------------------------------------- storage
def _load_json() -> dict:
    p = config.VEHICLE_REGISTRY_JSON
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _write_json(all_records: dict) -> None:
    try:
        config.VEHICLE_REGISTRY_JSON.write_text(
            json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


class DemoRegistryProvider:
    """Synthetic registry backed by SQLite (+ a JSON mirror). Records are permanent
    and only change on update()."""

    name = "demo"

    def get(self, plate: str) -> dict | None:
        norm = _norm(plate)
        if not norm:
            return None
        with database.get_conn() as conn:
            row = conn.execute("SELECT data FROM vehicle_registry WHERE plate=?", (norm,)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["data"])
        except Exception:
            return None

    def get_or_create(self, plate: str, hint: str | None = None) -> dict | None:
        norm = _norm(plate)
        if not norm:
            return None
        existing = self.get(norm)
        if existing is not None:
            return existing                      # NEVER regenerate an existing record
        rec = generate_record(norm, hint)
        now = database._now()
        with database.get_conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO vehicle_registry (plate, data, created_at, updated_at) "
                "VALUES (?,?,?,?)", (norm, json.dumps(rec), now, now))
        # re-read (covers the race where another call inserted first)
        rec = self.get(norm) or rec
        allj = _load_json(); allj[norm] = rec; _write_json(allj)
        return rec

    def update(self, plate: str, updates: dict) -> dict | None:
        norm = _norm(plate)
        rec = self.get(norm)
        if rec is None:
            rec = self.get_or_create(norm)
        if rec is None:
            return None
        # overwrite provided fields, but keep the identity keys stable
        for k, v in (updates or {}).items():
            if k in ("plate_normalized",):
                continue
            rec[k] = v
        rec["edited"] = True
        with database.get_conn() as conn:
            conn.execute("UPDATE vehicle_registry SET data=?, updated_at=? WHERE plate=?",
                         (json.dumps(rec), database._now(), norm))
        allj = _load_json(); allj[norm] = rec; _write_json(allj)
        return rec

    def list_all(self) -> list[dict]:
        with database.get_conn() as conn:
            rows = conn.execute("SELECT data FROM vehicle_registry ORDER BY created_at").fetchall()
        out = []
        for r in rows:
            try:
                out.append(json.loads(r["data"]))
            except Exception:
                pass
        return out


_provider: DemoRegistryProvider | None = None


def get_provider():
    """Return the active registry provider (swap here for a real police API)."""
    global _provider
    if _provider is None:
        _provider = DemoRegistryProvider()      # future: if config.REGISTRY_PROVIDER == "police_api": ...
    return _provider


def ensure(plate: str, hint: str | None = None) -> None:
    """Ingestion-time hook: make sure a record exists for a recognised plate.
    Never raises - a registry hiccup must not affect ingestion."""
    try:
        if plate:
            get_provider().get_or_create(plate, hint)
    except Exception:
        pass


def backfill_from_db() -> dict:
    """Create registry records for every plate already stored in the plates table
    (does not overwrite existing records). Used to seed the demo from prior ingests."""
    prov = get_provider()
    created = 0
    seen = set()
    with database.get_conn() as conn:
        rows = conn.execute(
            "SELECT p.plate_text AS t, d.class_label AS c FROM plates p "
            "LEFT JOIN detections d ON d.detection_id = p.detection_id").fetchall()
    for r in rows:
        norm = _norm(r["t"])
        if not norm or norm in seen:
            continue
        seen.add(norm)
        if prov.get(norm) is None:
            prov.get_or_create(norm, r["c"])
            created += 1
    return {"unique_plates": len(seen), "records_created": created,
            "total_records": len(prov.list_all())}


if __name__ == "__main__":
    for pl, h in [("GJ23EB1224", "motorcycle"), ("MH12AB1234", "car"), ("DL7CN1163", "auto-rickshaw")]:
        r = get_provider().get_or_create(pl, h)
        print(f"\n{pl} ({h}):")
        for k in ("owner_name", "vehicle_type", "vehicle_brand", "vehicle_model",
                  "registration_state", "registration_office", "insurance_status",
                  "stolen_status", "rc_status"):
            print(f"   {k:22s}: {r[k]}")
