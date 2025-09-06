import re
import cv2
import numpy as np
import easyocr
from pathlib import Path
from datetime import datetime, timedelta
from PIL import Image
from ultralytics import YOLO
import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from typing import Union
from supabase import create_client, Client

# --- Load Environment Variables ---
load_dotenv()

# --- Initialize Flask & Supabase ---
app = Flask(__name__)
CORS(app)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY]):
    print("FATAL: Supabase URL or Key not found in .env file.")
    exit()

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Successfully connected to Supabase.")
except Exception as e:
    print(f"FATAL: Could not connect to Supabase. Error: {e}")
    exit()


# ---------------- CONFIG ---------------- #
BASE_DIR = Path(__file__).parent
VEHICLE_MODEL_PATH = BASE_DIR / "models" / "vehicle_model.pt"
PLATE_MODEL_PATH = BASE_DIR / "models" / "license_plate_model.pt"

# Get email credentials from environment variables
ALERT_EMAIL = os.getenv("ALERT_EMAIL")
ALERT_RECIPIENT = os.getenv("ALERT_RECIPIENT")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")

# Setup paths for logs and uploads
LOGS_PATH = BASE_DIR / "logs"
LOGS_PATH.mkdir(parents=True, exist_ok=True)
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
BACKEND_LOG = LOGS_PATH / "backend.log"

# ---------------- LOGGER ---------------- #
logging.basicConfig(filename=str(BACKEND_LOG), level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
def log_event(msg):
    logging.info(msg)
    print(msg)

# ---------------- MODELS & OCR ---------------- #
try:
    vehicle_model = YOLO(str(VEHICLE_MODEL_PATH))
    plate_model = YOLO(str(PLATE_MODEL_PATH))
    reader = easyocr.Reader(['en'])
    log_event("Models and OCR reader loaded successfully.")
except Exception as e:
    log_event(f"FATAL: Could not load models. Error: {e}")
    vehicle_model = plate_model = reader = None

# ---------------- SHARED LOGIC ---------------- #
WHEELER_MAP = {"car": "4-wheeler", "bus": "4-wheeler+", "truck": "4-wheeler+", "auto": "3-wheeler", "bike": "2-wheeler", "ambulance": "4-wheeler+", "fire_truck": "4-wheeler+", "police": "4-wheeler+"}
TOLL_RATES = {"2-wheeler":20, "3-wheeler":30, "4-wheeler":50, "4-wheeler+":70}
PLATE_REGEX = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}[1-9]\d{0,3}$')

def is_valid_plate(plate):
    if plate is None: return False
    return bool(PLATE_REGEX.match(plate.strip().upper()))

def send_alert_email(plate, vtype):
    if not all([ALERT_EMAIL, EMAIL_APP_PASSWORD, ALERT_RECIPIENT]):
        log_event("Email credentials not configured in .env file. Skipping alert.")
        return
    try:
        msg = MIMEMultipart()
        msg["From"] = ALERT_EMAIL
        msg["To"] = ALERT_RECIPIENT
        msg["Subject"] = f"Stolen Vehicle Alert - {plate}"
        body = f"ALERT: Stolen Vehicle Detected!\nPlate: {plate}\nVehicle Type: {vtype}\nTime: {datetime.now()}"
        msg.attach(MIMEText(body, "plain"))
        context = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls(context=context)
            server.login(ALERT_EMAIL, EMAIL_APP_PASSWORD)
            server.send_message(msg)
        log_event(f"Sent stolen alert email for {plate}")
    except Exception as e:
        log_event(f"Failed to send alert email: {e}")

# ---------------- DATABASE FUNCTIONS ---------------- #
def is_vehicle_stolen(plate_number: str) -> bool:
    try:
        response = supabase.table('stolen_vehicles').select('id', count='exact').eq('number_plate', plate_number.upper()).execute()
        return response.count > 0
    except Exception as e:
        log_event(f"DB ERROR (is_vehicle_stolen): {e}")
        return False

def get_vehicle_info(plate_number: str) -> dict:
    try:
        response = supabase.table('vehicles').select('*').eq('number_plate', plate_number.upper()).limit(1).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        log_event(f"DB ERROR (get_vehicle_info): {e}")
        return None

def log_transaction(plate: str, vtype: str, amount: int, status: str, upi_id: str = None):
    try:
        supabase.table('payment_logs').insert({
            'plate_number': plate, 'vehicle_type': vtype, 'amount': amount,
            'status': status, 'upi_id': upi_id
        }).execute()
    except Exception as e:
        log_event(f"DB ERROR (log_transaction): {e}")

def was_scanned_within_12h(plate: str):
    try:
        twelve_hours_ago = (datetime.now() - timedelta(hours=12)).isoformat()
        response = supabase.table('payment_logs').select('id', count='exact').eq('plate_number', plate.upper()).gte('timestamp', twelve_hours_ago).execute()
        return response.count > 0
    except Exception as e:
        log_event(f"DB ERROR (was_scanned_within_12h): {e}")
        return False

def ocr_read_plate_from_crop(crop_path: Path):
    img = Image.open(str(crop_path)).convert('L')
    img_np = np.array(img)
    res = reader.readtext(img_np)
    if not res: return None
    text = "".join([r[1] for r in res]).upper().replace(" ", "")
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text.strip()

# ---------------- CORE PROCESSING ---------------- #
def process_vehicle(image_path: Union[str, Path]):
    image_path = str(image_path)
    try:
        vres = vehicle_model(image_path, imgsz=640, conf=0.4, verbose=False)
    except Exception as e:
        log_event(f"[ERROR] Vehicle model failed: {e}")
        return {"status":"error","msg":"Vehicle model failed"}

    if not vres or len(vres[0].boxes) == 0:
        return {"status":"no_vehicle","msg":"No vehicle detected"}

    vbox = vres[0].boxes[0]
    vcls = int(vbox.cls.cpu().item())
    vlabel = vres[0].names[vcls].lower()
    wheeler = WHEELER_MAP.get(vlabel, "unknown")
    log_event(f"Detected: {vlabel} ({wheeler})")

    if vlabel in ['ambulance','fire_truck','police']:
        log_transaction(None, vlabel, 0, "free_pass")
        log_event(f"[EMERGENCY] {vlabel.upper()} detected — FREE PASS")
        return {"status":"emergency", "vehicle_type":vlabel, "msg":"Free pass - gate opened"}

    try:
        pres = plate_model(image_path, imgsz=640, conf=0.4, verbose=False)
    except Exception as e:
        log_event(f"[ERROR] Plate model failed: {e}")
        return {"status":"error","msg":"Plate model failed"}

    if not pres or len(pres[0].boxes) == 0:
        log_event(f"{vlabel} detected but license plate not found")
        return {"status":"no_plate","vehicle_type":vlabel,"msg":"Vehicle detected but license plate not found"}

    pbox = pres[0].boxes[0]
    x1,y1,x2,y2 = [int(x) for x in pbox.xyxy[0].tolist()]
    temp_crop = LOGS_PATH / "temp_plate.jpg"
    Image.open(image_path).crop((x1,y1,x2,y2)).save(temp_crop)

    plate_text = ocr_read_plate_from_crop(temp_crop)
    if not plate_text:
        log_event(f"{vlabel} detected but OCR failed")
        return {"status":"ocr_failed","vehicle_type":vlabel,"msg":"OCR failed"}

    if not is_valid_plate(plate_text):
        log_event(f"OCR result '{plate_text}' invalid format — ignored")
        return {"status":"invalid_plate_format","vehicle_type":vlabel,"plate":plate_text,"msg":"Invalid plate format"}

    log_event(f"Plate recognized: {plate_text} | Vehicle: {vlabel}")

    if is_vehicle_stolen(plate_text):
        send_alert_email(plate_text, vlabel)
        log_transaction(plate_text, vlabel, 0, "stolen_alert")
        log_event(f"[ALERT] STOLEN VEHICLE {plate_text} — authorities alerted")
        return {"status":"stolen", "plate":plate_text, "vehicle_type":vlabel, "msg":"Stolen vehicle - authorities alerted"}

    if was_scanned_within_12h(plate_text):
        log_event(f"{plate_text} scanned within last 12 hours — no deduction")
        log_transaction(plate_text, vlabel, 0, "no_charge_recent")
        return {"status":"no_charge_recent", "plate":plate_text, "vehicle_type":vlabel, "msg":"Recently scanned — no deduction"}

    acct = get_vehicle_info(plate_text)
    if not acct:
        log_event(f"{plate_text} not registered for UPI")
        log_transaction(plate_text, vlabel, 0, "no_upi")
        return {"status":"no_upi", "plate":plate_text, "vehicle_type":vlabel, "msg":"No linked UPI — manual payment"}

    fee = TOLL_RATES.get(wheeler, 50)
    log_event(f"Deducting ₹{fee} from UPI {acct.get('upi_id')} for plate {plate_text}")
    log_transaction(plate_text, vlabel, fee, "paid", acct.get('upi_id'))

    return {"status":"paid", "plate":plate_text, "vehicle_type":vlabel, "amount":fee, "upi":acct.get('upi_id')}

# ---------------- API ENDPOINT ---------------- #
@app.route('/analyze', methods=['POST'])
def analyze_image():
    if not all([vehicle_model, plate_model, reader]):
        return jsonify({"status": "error", "msg": "Backend models are not loaded. Check server logs."}), 500
        
    if 'file' not in request.files:
        return jsonify({"status": "error", "msg": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "msg": "No file selected"}), 400

    filepath = None
    if file:
        try:
            filepath = UPLOAD_FOLDER / file.filename
            file.save(str(filepath))
            result = process_vehicle(filepath)
            os.remove(filepath)
            return jsonify(result)
        except Exception as e:
            log_event(f"Error during analysis: {e}")
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"status": "error", "msg": "An internal server error occurred."}), 500

# This allows the script to be run for development
if __name__ == '__main__':
    app.run(debug=True, port=5001)