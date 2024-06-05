import sys
import time
import subprocess

time.sleep(3)

if len(sys.argv)>1 and sys.argv[1] == 'restart':
    subprocess.Popen('streamlit run finetuning.py --server.headless true', shell=True)
else:
    subprocess.Popen('streamlit run finetuning.py', shell=True)
