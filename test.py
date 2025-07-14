from ultralytics import YOLO
import cv2
import pyttsx3
import time
from collections import Counter

model = YOLO("yolo11n.pt")

engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Use non-blocking mode
engine.startLoop(False)

last_time = 0
interval = 3

print("Starting real-time detection... Press 'q' to quit")
print("=" * 50)

for result in model(source=0, stream=True, conf=0.45):
    frame = result.plot()
    cv2.imshow("YOLOv8 Real-time Detection", frame)


    engine.iterate()

    if result.boxes is not None and len(result.boxes) > 0:
        labels = [model.names[int(cls)] for cls in result.boxes.cls.cpu().numpy()]

        if labels:
            counts = Counter(labels)

            now = time.time()
            if now - last_time > interval:
                # Get top 2 most common objects
                most_common = counts.most_common(2)


                speech_parts = []
                for label, count in most_common:
                    if count > 1:
                        speech_parts.append(f"{count} {label}s")  # Plural
                    else:
                        speech_parts.append(f"one {label}")


                if len(speech_parts) == 1:
                    speech = f"I see {speech_parts[0]}"
                else:
                    speech = f"I see {speech_parts[0]} and {speech_parts[1]}"


                print(f"\n[{time.strftime('%H:%M:%S')}] Detection results:")
                for label, count in counts.most_common():
                    print(f"  - {label}: {count}")
                print(f"Speaking: {speech}")
                print("-" * 50)


                engine.say(speech)

                last_time = now

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
engine.endLoop()
cv2.destroyAllWindows()
print("Program exited")