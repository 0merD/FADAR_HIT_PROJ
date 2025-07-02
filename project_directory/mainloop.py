import subprocess
import time
import sys
import os


FALL_DETECTION_SCRIPT = './pose_estimation_project.py'
NAVIGATION_SCRIPT='./yolobboxnavi.py'
HELPER_SCRIPT='./helperson.py'
BACKTOHOME_SCRIPT='./backtohome.py'
# Define the input argument for the fall detection script
INPUT_ARG = '--input'
INPUT_VALUE = 'rpi'

# Define the directory and file for fall detection logs can be used for future development
LOG_DIR = "./datadir"
LOG_FILE = os.path.join(LOG_DIR, "fall_detections.txt")

def read_last_fall_event(log_file_path):
    """
    Reads the last line (most recent fall event) from the specified log file.
    Returns the last line as a string, or None if the file is empty or doesn't exist.
    """
    if not os.path.exists(log_file_path):
        return None
    
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
            if lines:
                return lines[-1].strip() # Get the last non-empty line and remove whitespace
            else:
                return None
    except Exception as e:
        print(f"Error reading log file '{log_file_path}': {e}", file=sys.stderr)
        return None

def run_detector_subprocess():
    """
    Runs the fall detection script as a subprocess and then processes its log file.
    """
    print(f"Starting fall detection script: {FALL_DETECTION_SCRIPT} {INPUT_ARG} {INPUT_VALUE}")
    
    try:
        # Use subprocess.run to execute the script.
        # It will block until the subprocess finishes.
        process = subprocess.run(
            [sys.executable, FALL_DETECTION_SCRIPT, INPUT_ARG, INPUT_VALUE],
            check=False, # Set to False so we can check returncode manually
            capture_output=False
        )
        
        # Check the return code of the subprocess
        if process.returncode == 0:
            print("\nFall detection script finished gracefully.")
        else:
            print(f"\nFall detection script exited with non-zero status code: {process.returncode}.", file=sys.stderr)

        # After the subprocess finishes, read the last fall event
        print(f"\nChecking for fall detection logs in: {LOG_FILE}")
        last_event = read_last_fall_event(LOG_FILE)

        if last_event:
            print("\n--- Last Fall Event Detected ---")
            print(last_event)
            print("--------------------------------")
        else:
            print("No fall events recorded or log file is empty.")

    except FileNotFoundError:
        print(f"Error: Script not found at '{FALL_DETECTION_SCRIPT}'. Please ensure the path is correct.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nMain script received Ctrl+C. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred in the main script: {e}", file=sys.stderr)
        sys.exit(1)

def run_navigation_subprocess():
    """
    Runs the navigation script as a subprocess.
    """
    print(f"Starting navigation: {NAVIGATION_SCRIPT} {INPUT_ARG} {INPUT_VALUE}")
    
    try:
        # Use subprocess.run to execute the script.
        # It will block until the subprocess finishes.
        process = subprocess.run(
            [sys.executable, NAVIGATION_SCRIPT, INPUT_ARG, INPUT_VALUE],
            check=False, # Set to False so we can check returncode manually
            capture_output=False
        )
        
        # Check the return code of the subprocess
        if process.returncode == 0:
            print("\nNavigation script finished gracefully.")
        else:
            print(f"\nNavigation script exited with non-zero status code: {process.returncode}.", file=sys.stderr)

        # After the subprocess finishes, read the last fall event
        print(f"\nAT TARGET PERSON")
        
    except KeyboardInterrupt:
        print("\nMain script received Ctrl+C. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred in the main script: {e}", file=sys.stderr)
        sys.exit(1)

def run_helper_subprocess():
    """
    Runs the helper script as a subprocess.
    """
    print(f"HELPING: {HELPER_SCRIPT}")
    
    try:
        # Use subprocess.run to execute the script.
        # It will block until the subprocess finishes.
        process = subprocess.run(
            [sys.executable, HELPER_SCRIPT],
            check=False, # Set to False so we can check returncode manually
            capture_output=False
        )
        
        # Check the return code of the subprocess
        if process.returncode == 0:
            print("\nHelper script finished gracefully.")
        else:
            print(f"\nHelper script exited with non-zero status code: {process.returncode}.", file=sys.stderr)

        # After the subprocess finishes, read the last fall event
        print(f"\nNext Iteration")
        
    except KeyboardInterrupt:
        print("\nMain script received Ctrl+C. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred in the main script: {e}", file=sys.stderr)
        sys.exit(1)

def run_backtohome_subprocess():
    """
    Runs the backtohome script as a subprocess.
    """
    print(f"Going Back to Home: {BACKTOHOME_SCRIPT}")
    
    try:
        # Use subprocess.run to execute the script.
        # It will block until the subprocess finishes.
        process = subprocess.run(
            [sys.executable, BACKTOHOME_SCRIPT],
            check=False, # Set to False so we can check returncode manually
            capture_output=False
        )
        
        # Check the return code of the subprocess
        if process.returncode == 0:
            print("\nbacktohome script finished gracefully.")
        else:
            print(f"\backtohome script exited with non-zero status code: {process.returncode}.", file=sys.stderr)

        # After the subprocess finishes, read the last fall event
        print(f"\nNext Iteration")
        
    except KeyboardInterrupt:
        print("\nMain script received Ctrl+C. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred in the main script: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    while(True):
        run_detector_subprocess()
        run_navigation_subprocess()
        run_helper_subprocess()
        run_backtohome_subprocess()