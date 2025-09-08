import subprocess
import sys
from pathlib import Path

# Create logs directory
Path("logs").mkdir(exist_ok=True)

# Scripts to run in order
scripts = [
    ("mobilenet.py", "logs/mobilenet.log"),
    ("vgg.py", "logs/vgg.log"),
    ("resnet.py", "logs/resnet.log")
]

for script, log_file in scripts:
    print(f"\n🚀 Running {script}...")
    with open(log_file, "w") as log:
        process = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in process.stdout:
            print(line, end='')   # Show in console
            log.write(line)       # Save to log

        process.wait()

        if process.returncode != 0:
            print(f"❌ {script} failed with exit code {process.returncode}. Stopping execution.")
            sys.exit(process.returncode)

    print(f"✅ Finished {script}. Logs saved to {log_file}")

print("\n🎉 All models executed successfully.")
