import time
from gpiozero import OutputDevice
import sys
import os
import threading # Ensure threading is imported for DualStepperMotorController

# --- Dual Stepper Motor Control Class (Copied from your original script) ---
class DualStepperMotorController:
    def __init__(self,
                 left_in1_pin, left_in2_pin, left_in3_pin, left_in4_pin,
                 right_in1_pin, right_in2_pin, right_in3_pin, right_in4_pin,
                 min_delay):
        self.min_delay = min_delay

        self.left_pins = [OutputDevice(left_in1_pin), OutputDevice(left_in2_pin),
                          OutputDevice(left_in3_pin), OutputDevice(left_in4_pin)]
        self.right_pins = [OutputDevice(right_in1_pin), OutputDevice(right_in2_pin),
                           OutputDevice(right_in3_pin), OutputDevice(right_in4_pin)]

        self._running_left = threading.Event()
        self._running_right = threading.Event()
        self._direction_left = 0
        self._direction_right = 0
        self._left_motor_thread = None
        self._right_motor_thread = None

        self._reset_gpio_all()

    def _reset_gpio_all(self):
        """Sets all motor control pins to LOW for both motors."""
        for pin in self.left_pins + self.right_pins:
            pin.off()

    def _run_single_stepper_motor(self, pins, motor_name, direction, running_event):
        """
        Internal function to drive a single 28BYJ-48 stepper motor.
        Designed to be run as a separate thread.
        """
        if direction == 0:
            print(f"Direction = 0: {motor_name} motor will not run.")
            running_event.clear()
            return

        in1, in2, in3, in4 = pins

        # 8-step sequence for 28BYJ-48 motor
        base_sequence = [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 1],
        ]

        # Determine the step sequence based on motor_name and desired direction
        # Note: Directions are reversed for backward movement (direction=1 for robot forward, -1 for robot backward)
        # So, for the robot to move backward, left_direction will be -1 and right_direction will be -1.
        # This means the Left motor needs reversed(base_sequence) and Right motor needs base_sequence.
        if motor_name == "Left":
            if direction == -1:  # Left motor "forward" (e.g., moves robot forward)
                step_sequence = base_sequence
            else:  # Left motor "backward" (e.g., moves robot backward) - this is what we want for robot reverse
                step_sequence = list(reversed(base_sequence))
        elif motor_name == "Right":
            # For the right motor, 'forward' for the robot might mean spinning
            # 'backwards' relative to its own physical setup to achieve mirrored movement.
            if direction == -1:  # Right motor "forward" (for mirrored wheel movement) - this is what we want for robot reverse
                step_sequence = list(reversed(base_sequence))
            else:  # Right motor "backward" (for mirrored wheel movement)
                step_sequence = base_sequence

        step_number = 0
        delay = 0.01  # Initial delay for ramp-up
        ramp_rate = 0.0001 # How quickly to reduce delay

        while running_event.is_set():
            current_step_pattern = step_sequence[step_number % len(step_sequence)]
            in1.value, in2.value, in3.value, in4.value = current_step_pattern

            step_number += 1
            time.sleep(delay)

            # Implement ramp-up to min_delay
            if delay > self.min_delay:
                delay -= ramp_rate
            else:
                delay = self.min_delay # Ensure it doesn't go below min_delay

        # Ensure pins are off when the thread stops
        for pin in pins:
            pin.off()
        print(f"{motor_name} motor loop stopped and pins reset.")


    def start(self, left_direction=0, right_direction=0):
        """
        Starts one or both motors in the specified directions.
        Directions: 1 for robot forward, -1 for robot backward, 0 for stop.
        """
        print(f"Attempting to start motors: Left Direction: {left_direction}, Right Direction: {right_direction}")

        # Handle Left Motor
        if left_direction != self._direction_left or not self._running_left.is_set():
            if self._left_motor_thread and self._left_motor_thread.is_alive():
                self._running_left.clear() # Signal current thread to stop
                self._left_motor_thread.join(timeout=0.5) # Wait briefly for it to stop
                if self._left_motor_thread.is_alive():
                    print("Warning: Left motor thread did not terminate gracefully.")

            self._direction_left = left_direction
            if left_direction != 0:
                self._running_left.set()
                self._left_motor_thread = threading.Thread(
                    target=self._run_single_stepper_motor,
                    args=(self.left_pins, "Left", self._direction_left, self._running_left)
                )
                self._left_motor_thread.daemon = True
                self._left_motor_thread.start()
                print(f"Left motor started in direction: {left_direction}")
            else:
                print("Left motor commanded to stop.")
                # Ensure pins are off if stopping
                for pin in self.left_pins:
                    pin.off()


        # Handle Right Motor
        if right_direction != self._direction_right or not self._running_right.is_set():
            if self._right_motor_thread and self._right_motor_thread.is_alive():
                self._running_right.clear() # Signal current thread to stop
                self._right_motor_thread.join(timeout=0.5) # Wait briefly for it to stop
                if self._right_motor_thread.is_alive():
                    print("Warning: Right motor thread did not terminate gracefully.")

            self._direction_right = right_direction
            if right_direction != 0:
                self._running_right.set()
                self._right_motor_thread = threading.Thread(
                    target=self._run_single_stepper_motor,
                    args=(self.right_pins, "Right", self._direction_right, self._running_right)
                )
                self._right_motor_thread.daemon = True
                self._right_motor_thread.start()
                print(f"Right motor started in direction: {right_direction}")
            else:
                print("Right motor commanded to stop.")
                # Ensure pins are off if stopping
                for pin in self.right_pins:
                    pin.off()

    def stop_all(self):
        """Stops both motors and their associated threads."""
        print("Stopping all motors...")
        if self._running_left.is_set():
            self._running_left.clear()
            if self._left_motor_thread and self._left_motor_thread.is_alive():
                self._left_motor_thread.join(timeout=1)
        if self._running_right.is_set():
            self._running_right.clear()
            if self._right_motor_thread and self._right_motor_thread.is_alive():
                self._right_motor_thread.join(timeout=1)

        self._direction_left = 0
        self._direction_right = 0
        self._reset_gpio_all()
        print("All stepper motors stopped.")

    def cleanup(self):
        """Cleans up gpiozero objects by ensuring all pins are off."""
        self.stop_all()
        print("GPIO Zero objects handled (all pins set to off).")

# --- End of Copied Class ---

def read_movement_duration(log_file_path="movementlog"):
    """
    Reads the movement duration from the specified log file.
    """
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found at {log_file_path}", file=sys.stderr)
        return None
    try:
        with open(log_file_path, "r") as f:
            duration_str = f.read().strip()
            duration = float(duration_str)
            print(f"Read movement duration: {duration} seconds from {log_file_path}")
            return duration
    except ValueError:
        print(f"Error: Could not parse duration from {log_file_path}. Is it a valid number?", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading log file: {e}", file=sys.stderr)
        return None

def clear_movement_log(log_file_path="movementlog"):
    """
    Clears the content of the movement log file.
    If the file doesn't exist, it creates an empty one.
    """
    try:
        # 'w' mode truncates the file (clears its content) or creates it if it doesn't exist.
        with open(log_file_path, "w") as f:
            f.write("")
        print(f"Cleared content of {log_file_path}")
    except Exception as e:
        print(f"Error clearing movement log file {log_file_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    # Define the path to your movementlog file.
    # It assumes 'movementlog' is in the same directory as this script.
    LOG_FILE = "movementlog"

    # Read the duration
    movement_duration = read_movement_duration(LOG_FILE)

    if movement_duration is None:
        print("Exiting. Could not determine reverse movement duration.")
        sys.exit(1)

    # Initialize the DualStepperMotorController with your specific pin numbers.
    # IMPORTANT: Ensure these pin numbers match your physical wiring exactly.
    motor_controller = DualStepperMotorController(
        left_in1_pin=17, left_in2_pin=27, left_in3_pin=22, left_in4_pin=24, # Example pins for Left Motor
        right_in1_pin=5, right_in2_pin=6, right_in3_pin=13, right_in4_pin=19, # Example pins for Right Motor
        min_delay=0.0015 # Use the same min_delay for consistent motor speed
    )

    try:
        print(f"Starting robot reverse movement for {movement_duration:.2f} seconds...")
        # Start motors backward:
        # Based on your previous script, robot forward is left_direction=1, right_direction=1.
        # So, robot backward would be left_direction=-1, right_direction=-1.
        motor_controller.start(left_direction=-1, right_direction=-1)

        # Wait for the specified duration
        time.sleep(movement_duration)

        print("Reverse movement complete. Stopping motors.")
        motor_controller.stop_all()

    except KeyboardInterrupt:
        print("\nReverse movement interrupted by user.")
    except Exception as e:
        print(f"An error occurred during reverse movement: {e}", file=sys.stderr)
    finally:
        motor_controller.cleanup()
        # --- Clear the movementlog file ---
        clear_movement_log(LOG_FILE)
        print("Reverse movement script finished.")