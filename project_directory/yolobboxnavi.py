import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from datetime import datetime
import numpy as np
import cv2
import hailo
import sys
import time
import threading
from gpiozero import OutputDevice
import os

# --- Dual Stepper Motor Control Class ---
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
        # Adjust these conditions if your physical setup requires different mirroring
        if motor_name == "Left":
            if direction == -1:  # Left motor "forward" (e.g., moves robot forward)
                step_sequence = base_sequence
            else:  # Left motor "backward" (e.g., moves robot backward)
                step_sequence = list(reversed(base_sequence))
        elif motor_name == "Right":
            # For the right motor, 'forward' for the robot might mean spinning
            # 'backwards' relative to its own physical setup to achieve mirrored movement.
            if direction == -1:  # Right motor "forward" (for mirrored wheel movement)
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


from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp


class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42
        self.target_person_id = None
        self.frame_width = 0
        self.frame_height = 0
        self.is_centered = False
        self.motor_controller = None # DualStepperMotorController
        self.movement_start_time = None # Added for logging movement time
        self.movement_logging_enabled = False # Added to control logging


def attempt_app_shutdown_async(app_instance):
    """
    Called asynchronously from GLib's idle loop to attempt to shut down the app.
    It first tries to call app_instance.stop(), then falls back to forceful exit.
    """
    print("Executing asynchronous app shutdown attempt...")

    if hasattr(app_instance, 'stop') and callable(app_instance.stop):
        print("Attempting to call app_instance.stop() for graceful shutdown.")
        try:
            app_instance.stop()
            print("app_instance.stop() called. Waiting for app to exit.")
        except Exception as e:
            print(f"Error calling app_instance.stop(): {e}. Resorting to forceful exit.", file=sys.stderr)
            sys.exit(0)
    else:
        print("app_instance does not have a callable 'stop' method. Falling back to pipeline NULL state and forceful exit if needed.", file=sys.stderr)
        pipeline = None
        if hasattr(app_instance, '_GstApp__pipeline'):
             pipeline = getattr(app_instance, '_GstApp__pipeline')
        elif hasattr(app_instance, 'pipeline'):
             pipeline = getattr(app_instance, 'pipeline')

        if pipeline:
            try:
                pipeline.set_state(Gst.State.NULL)
                print("GStreamer pipeline state to NULL set.")
                print("Pipeline set to NULL. If app remains stuck, it might require forceful termination.")
                sys.exit(0)
            except Exception as e:
                print(f"Error setting pipeline state to NULL: {e}. Resorting to forceful exit.", file=sys.stderr)
                sys.exit(0)
        else:
            print("Warning: Could not find pipeline object in GStreamerDetectionApp instance. Resorting to forceful exit.", file=sys.stderr)
            sys.exit(0)

    return GLib.SOURCE_REMOVE


def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    format, width, height = get_caps_from_pad(pad)
    if user_data.frame_width == 0 and width is not None:
        user_data.frame_width = width
    if user_data.frame_height == 0 and height is not None:
        user_data.frame_height = height

    frame = None
    if user_data.use_frame and format and width and height:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    detection_count = 0
    current_target_bbox_height = 0
    target_found_in_frame = False

    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()

        if label == "person":
            track_id = 0
            track_objects = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track_objects) == 1:
                track_id = track_objects[0].get_id()

            if user_data.target_person_id is None:
                user_data.target_person_id = track_id
                user_data.is_centered = False
                string_to_print += f"Target acquired: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n"

            if user_data.target_person_id == track_id:
                target_found_in_frame = True
                x_min = int(bbox.xmin() * width)
                y_min = int(bbox.ymin() * height)
                x_max = int((bbox.xmin() + bbox.width()) * width)
                y_max = int((bbox.ymin() + bbox.height()) * height)
                bbox_center_x = (x_min + x_max) // 2
                current_target_bbox_height = y_max - y_min

                if user_data.use_frame:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                    cv2.putText(frame, f"Target: ID {track_id}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                frame_center_x = width // 2
                centering_margin = width * 0.1

                # --- Motor Control Logic for Centering (Dual Motor) ---
                if not user_data.is_centered:
                    if bbox_center_x < frame_center_x - centering_margin:
                        string_to_print += "Moving robot left: Target too far left.\n"
                        if user_data.motor_controller:
                            # Left motor moves forward, Right motor moves backward (robot turns left)
                            user_data.motor_controller.start(left_direction=1, right_direction=-1)
                            # Reset movement start time if turning, as forward movement hasn't truly started
                            user_data.movement_start_time = None
                            user_data.movement_logging_enabled = False
                    elif bbox_center_x > frame_center_x + centering_margin:
                        string_to_print += "Moving robot right: Target too far right.\n"
                        if user_data.motor_controller:
                            # Left motor moves backward, Right motor moves forward (robot turns right)
                            user_data.motor_controller.start(left_direction=-1, right_direction=1)
                            # Reset movement start time if turning
                            user_data.movement_start_time = None
                            user_data.movement_logging_enabled = False
                    else:
                        user_data.is_centered = True
                        string_to_print += "Target is now centered horizontally. Stopping motors to check height.\n"
                        if user_data.motor_controller:
                            user_data.motor_controller.stop_all()
                            # Reset movement start time as horizontal centering is done
                            user_data.movement_start_time = None
                            user_data.movement_logging_enabled = False
                else: # Person is centered horizontally, now check height
                    # If centered, move robot forward until program finishes (height condition met)
                    if current_target_bbox_height < user_data.frame_height * 0.95:
                        string_to_print += "Target centered but not close enough. Moving robot forward.\n"
                        if user_data.motor_controller:
                            # Check if motors are not already moving forward to avoid resetting time unnecessarily
                            if not (user_data.motor_controller._direction_left == 1 and
                                    user_data.motor_controller._direction_right == 1 and
                                    user_data.motor_controller._running_left.is_set() and
                                    user_data.motor_controller._running_right.is_set()):
                                user_data.motor_controller.start(left_direction=1, right_direction=1)
                                if user_data.movement_start_time is None:
                                    user_data.movement_start_time = time.monotonic()
                                    user_data.movement_logging_enabled = True
                    else:
                        string_to_print += "TARGET ARRIVED! Person is centered and close enough.\n"
                        if user_data.motor_controller:
                            user_data.motor_controller.stop_all() # Stop motors

                        # --- Log movement time ---
                        if user_data.movement_logging_enabled and user_data.movement_start_time is not None:
                            movement_duration = time.monotonic() - user_data.movement_start_time
                            try:
                                # This will create the file in the directory where the script is executed.
                                with open("movementlog", "w") as f:
                                    f.write(str(movement_duration))
                                print(f"Logged movement duration: {movement_duration} seconds to movementlog (in current directory)")
                            except IOError as e:
                                print(f"Error writing to movement log file: {e}", file=sys.stderr)
                            finally:
                                user_data.movement_logging_enabled = False
                                user_data.movement_start_time = None

                        GLib.idle_add(attempt_app_shutdown_async, user_data.app_instance)

            string_to_print += f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n"
            detection_count += 1

    # If the target person is lost (not detected in the current frame)
    if user_data.target_person_id is not None and not target_found_in_frame:
        string_to_print += "Target person lost. Treating as Target Arrived.\n"
        if user_data.motor_controller:
            user_data.motor_controller.stop_all()

        # --- Log movement time (same as TARGET ARRIVED) ---
        if user_data.movement_logging_enabled and user_data.movement_start_time is not None:
            movement_duration = time.monotonic() - user_data.movement_start_time
            try:
                with open("movementlog", "w") as f:
                    f.write(str(movement_duration))
                print(f"Logged movement duration: {movement_duration} seconds to movementlog (in current directory)")
            except IOError as e:
                print(f"Error writing to movement log file: {e}", file=sys.stderr)
            finally:
                user_data.movement_logging_enabled = False
                user_data.movement_start_time = None

        # --- Initiate shutdown (same as TARGET ARRIVED) ---
        GLib.idle_add(attempt_app_shutdown_async, user_data.app_instance)


    # If no detections at all or no person detected yet, ensure motors aren't running unexpectedly
    if detection_count == 0 and user_data.motor_controller:
        # Check if either motor is currently running
        if user_data.motor_controller._running_left.is_set() or user_data.motor_controller._running_right.is_set():
             string_to_print += "No detections. Stopping motors.\n"
             user_data.motor_controller.stop_all()
             user_data.movement_start_time = None
             user_data.movement_logging_enabled = False


    if user_data.use_frame and frame is not None:
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Is Centered: {user_data.is_centered}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Target ID: {user_data.target_person_id}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    user_data = user_app_callback_class()

    # Initialize the DualStepperMotorController with pins for both motors.
    # IMPORTANT: Adjust these pin numbers to match your actual wiring for Left and Right motors.
    motor_controller = DualStepperMotorController(
        left_in1_pin=17, left_in2_pin=27, left_in3_pin=22, left_in4_pin=24, # Example pins for Left Motor
        right_in1_pin=5, right_in2_pin=6, right_in3_pin=13, right_in4_pin=19, # Example pins for Right Motor
        min_delay=0.0015
    )
    user_data.motor_controller = motor_controller

    # Ensure motors are stopped at startup
    motor_controller.stop_all()

    app = GStreamerDetectionApp(app_callback, user_data)
    user_data.app_instance = app

    try:
        app.run()
        print("Application main loop started. Waiting for pipeline to finish or manual interrupt.")

    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down...")
    finally:
        pipeline = None
        if user_data.app_instance:
            if hasattr(user_data.app_instance, '_GstApp__pipeline'):
                pipeline = getattr(user_data.app_instance, '_GstApp__pipeline')
            elif hasattr(user_data.app_instance, 'pipeline'):
                pipeline = getattr(user_data.app_instance, 'pipeline')

        if pipeline and pipeline.get_state(Gst.State.NULL)[1] != Gst.State.NULL:
            print("Ensuring pipeline is set to NULL during final cleanup.")
            pipeline.set_state(Gst.State.NULL)

        motor_controller.cleanup()
        print("Script finished.")