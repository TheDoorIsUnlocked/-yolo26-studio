import os
import subprocess
import shutil

def build_exe():
    # Define build parameters
    script_path = "main.py"
    app_name = "YOLO26_Studio"
    
    # Clean previous builds
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    if os.path.exists(f"{app_name}.spec"):
        os.remove(f"{app_name}.spec")

    # PyInstaller arguments
    args = [
        "pyinstaller",
        "--noconfirm",
        "--clean",
        "--windowed",  # No console window
        f"--name={app_name}",
        # Add hidden imports often missed by PyInstaller for these libs
        "--hidden-import=ultralytics",
        "--hidden-import=PyQt6",
        "--hidden-import=PIL",
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        # Explicitly hide-import local modules just in case
        "--hidden-import=styles",
        "--hidden-import=config",
        "--hidden-import=workers",
        # Collect all ultralytics data (config, assets) to avoid runtime errors
        "--collect-all=ultralytics",
        "--collect-all=ultralytics.models",
        "--collect-all=ultralytics.nn",
        "--collect-all=ultralytics.utils",
        # Main script
        script_path
    ]

    print("Running PyInstaller...")
    print(" ".join(args))
    
    result = subprocess.run(args)
    
    if result.returncode == 0:
        print("\nBuild successful!")
        print(f"Executable is located in: dist\\{app_name}\\{app_name}.exe")
    else:
        print("\nBuild failed!")

if __name__ == "__main__":
    build_exe()
