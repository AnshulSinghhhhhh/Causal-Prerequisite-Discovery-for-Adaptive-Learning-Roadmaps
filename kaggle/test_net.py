"""Check internet access and environment on Kaggle."""
import socket
import sys

print("Python version:", sys.version)

try:
    s = socket.create_connection(("8.8.8.8", 53), timeout=3)
    print("[SUCCESS] Outbound network connection to 8.8.8.8 succeeded!")
    s.close()
except Exception as e:
    print("[FAILED] Outbound network connection failed:", e)

try:
    ip = socket.gethostbyname("huggingface.co")
    print("[SUCCESS] DNS resolution for huggingface.co succeeded:", ip)
except Exception as e:
    print("[FAILED] DNS resolution failed:", e)
