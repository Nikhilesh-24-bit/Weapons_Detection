import cv2
from ultralytics import YOLO
import csv
import os
from datetime import datetime
from telegram import Bot
import asyncio
from geopy.geocoders import Nominatim

# Telegram credentials
BOT_TOKEN = '8165682771:AAFdf2MeGjAG8cDfAmHpenJqdqsjCWjClsU'
CHAT_ID = '1313464419'

# Get camera setup from operator
def get_camera_details():
    print("\n🔧 Enter camera details:")
    cam_id = input("Camera ID (e.g., Cam1): ")
    room_name = input("Room/Location name (e.g., Entrance, Lab): ")
    latitude = float(input("Latitude (e.g., 21.9386): "))
    longitude = float(input("Longitude (e.g., 86.7461): "))
    return cam_id, room_name, latitude, longitude

# Reverse geocode to get address from lat/lon
def get_location_name(lat, lon):
    try:
        geolocator = Nominatim(user_agent="weapon-detector")
        location = geolocator.reverse((lat, lon), language='en')
        return location.address if location else "Unknown location"
    except Exception as e:
        print(f"[ERROR] Reverse geocoding failed: {e}")
        return "Unknown location"

# Async Telegram alert function
async def send_telegram_alert(image_path, class_name, confidence, cam_id, room_name, lat, lon):
    try:
        bot = Bot(token=BOT_TOKEN)
        location_name = get_location_name(lat, lon)
        map_link = f"https://maps.google.com/?q={lat},{lon}"

        caption = (
            f"\U0001F6A8 Weapon Detected!\n"
            f"Camera: {cam_id}\n"
            f"Room: {room_name}\n"
            f"Class: {class_name}\n"
            f"Confidence: {confidence}\n"
            f"\U0001F4CD Location: {location_name}\n"
            f"\U0001F310 Google Maps: {map_link}"
        )

        with open(image_path, 'rb') as photo:
            await bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=caption)

        await bot.send_location(chat_id=CHAT_ID, latitude=lat, longitude=lon)
        print("[INFO] Telegram alert sent.")
    except Exception as e:
        print(f"[ERROR] Telegram alert failed: {e}")

# Synchronous wrapper
def send_telegram_alert_sync(image_path, class_name, confidence, cam_id, room_name, lat, lon):
    asyncio.run(send_telegram_alert(image_path, class_name, confidence, cam_id, room_name, lat, lon))

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Get camera info
    CAMERA_ID, ROOM_NAME, LATITUDE, LONGITUDE = get_camera_details()

    # Load YOLOv8 model
    model = YOLO(r"C:\Users\nikhi\Desktop\weapon-detection\runs\detect\train5\weights\best.pt")

    cap = cv2.VideoCapture(0)

    # CSV setup
    log_file = 'detection_log.csv'
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Camera', 'Room', 'Class', 'Confidence', 'Latitude', 'Longitude', 'Address'])

    # Image folder
    image_save_dir = 'detected_images'
    os.makedirs(image_save_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=0.5, device=0, imgsz=320)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                conf = box.conf.item()

                if class_name in ['pistol', 'knife']:
                    weapon_box = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, weapon_box)

                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    # Save image
                    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    filename = f"{CAMERA_ID}_{class_name}_{timestamp}.jpg"
                    filepath = os.path.join(image_save_dir, filename)
                    cv2.imwrite(filepath, frame)

                    # Get location name
                    location_name = get_location_name(LATITUDE, LONGITUDE)

                    # Log to CSV
                    with open(log_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp, CAMERA_ID, ROOM_NAME, class_name, f'{conf:.2f}', LATITUDE, LONGITUDE, location_name])

                    # Send alert
                    send_telegram_alert_sync(filepath, class_name, f'{conf:.2f}', CAMERA_ID, ROOM_NAME, LATITUDE, LONGITUDE)

        # Display frame
        cv2.imshow('Weapon Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
