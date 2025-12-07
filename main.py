import os, time
from ultralytics import YOLO
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

def detect_accident(image_path: str) -> str:
    results = model.predict(image_path)
    accident_detected = any(model.names[int(box.cls[0])] == "accident" and float(box.conf[0]) > 0.5
                            for r in results for box in r.boxes)
    return "Accident detected!" if accident_detected else "No accident detected."

def send_alert(message: str):
    print(f"⚠️ ALERT: {message}")

accident_tool = Tool(name="AccidentDetectionTool", func=detect_accident,
                     description="Detects if an image contains a vehicle accident.")
llm = OpenAI(temperature=0)
agent = initialize_agent(tools=[accident_tool], llm=llm,
                         agent="zero-shot-react-description", verbose=True)

WATCH_FOLDER = "accident_images"
os.makedirs(WATCH_FOLDER, exist_ok=True)
seen_files = set()
print(f"Monitoring folder: {WATCH_FOLDER} for new images...")

while True:
    files = set(os.listdir(WATCH_FOLDER))
    new_files = files - seen_files
    for f in new_files:
        if f.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(WATCH_FOLDER, f)
            print(f"New image detected: {image_path}")
            result = detect_accident(image_path)
            print(result)
            if "Accident detected" in result:
                send_alert(f"Accident detected in {f}")
    seen_files = files
    time.sleep(5)

