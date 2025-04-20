import os
import subprocess
import sys

# app.py 파일 경로
current_file = os.path.abspath("app.py")

# streamlit으로 실행
subprocess.run([sys.executable, "-m", "streamlit", "run", current_file])
