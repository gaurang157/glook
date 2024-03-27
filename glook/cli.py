import os
import subprocess
# from .GView import main2

def main2():
    # Get the directory of the current module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "GLook.py")
    subprocess.call(["streamlit", "run", script_path])

if __name__ == "__main__":
    main2()
