"""Opens the live dashboard (served by serve.py)."""
import webbrowser
from config import API
webbrowser.open(API + "/"); print("Dashboard:", API + "/")
