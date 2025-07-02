import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import math
from datetime import datetime
import sys
import time
import threading
from gpiozero import OutputDevice

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
            # Ensure pins are off if this thread is told to stop
            for pin in pins:
                pin.off()
            return

        in1, in2, in3, in4 = pins

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
        # This is the logic you specified as correct:
        if motor_name == "Left":
            if direction == -1:  # Left motor forward
                step_sequence = base_sequence
            else:  # Left motor backward
                step_sequence = list(reversed(base_sequence))
        elif motor_name == "Right":
            if direction == -1:  # Right motor forward (for mirrored wheel movement)
                step_sequence = list(reversed(base_sequence))
            else:  # Right motor backward (for mirrored wheel movement)
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


# Define the directory for logs (must match the main script) fall logs are not necessery for current version to work and exist for debugging and future development
LOG_DIR = "./datadir"
LOG_FILE = os.path.join(LOG_DIR, "fall_detections.txt")

# Define the fall confirmation timer duration in seconds
FALL_TIMER_SECONDS = 5

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function 
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.app_instance = None
        self.fall_detected_and_stopping = False
        self.fall_detection_initiated = False 
        self.fall_start_time = None
        self.motor_controller = None # Will store the instance of DualStepperMotorController

# -----------------------------------------------------------------------------------------------
# User-defined callback function 
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    if user_data.fall_detected_and_stopping:
        return Gst.PadProbeReturn.REMOVE

    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    format, width, height = get_caps_from_pad(pad)
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    keypoints_map = get_keypoints()

    # Calculate frame center boundaries
    frame_center_x = width / 2
    left_bound = frame_center_x - (width / 4)
    right_bound = frame_center_x + (width / 4)
    string_to_print += f"  Horizontal center range: [{left_bound:.0f}, {right_bound:.0f}]\n"


    current_fall_condition_met = False # Flag to indicate if current frame meets fall+center criteria

    # Initialize bbox_center_x, bbox_center_y, track_id for logging, in case no person is detected
    bbox_center_x = -1
    bbox_center_y = -1
    track_id = -1

    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()

        if label == "person":
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()

            string_to_print += f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n"
            string_to_print += "  Lying Down Logic Details:\n"

            is_lying_down = False
            alert_reasons = []

            bbox_abs_pixel_width = bbox.width() * width
            bbox_abs_pixel_height = bbox.height() * height
            aspect_ratio_hw = -1.0

            degreethres=30
            
            # Define confidence threshold for keypoints
            KEYPOINT_CONFIDENCE_THRESHOLD = 0.8

            if bbox_abs_pixel_width > 0:
                aspect_ratio_hw = bbox_abs_pixel_height / bbox_abs_pixel_width
                string_to_print += f"    BBox: H={bbox_abs_pixel_height:.0f}, W={bbox_abs_pixel_width:.0f}, Ratio H/W={aspect_ratio_hw:.2f}\n"
                if aspect_ratio_hw < 0.84:
                    is_lying_down = True
                    alert_reasons.append("H<W")
                    string_to_print += "      Lying Trigger: BBox H<W\n"
                else:
                    string_to_print += "      BBox H<W: No\n"
            else:
                string_to_print += f"    BBox: H={bbox_abs_pixel_height:.0f}, W={bbox_abs_pixel_height:.0f} (Width is zero, check skipped)\n"

            landmarks_objects = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            # Store points and their confidences
            points_data = {}
            if len(landmarks_objects) != 0:
                landmarks = landmarks_objects[0].get_points()
                for i, point in enumerate(landmarks):
                    x_norm = point.x()
                    y_norm = point.y()
                    x_pixel = int((x_norm * bbox.width() + bbox.xmin()) * width)
                    y_pixel = int((y_norm * bbox.height() + bbox.ymin()) * height)
                    points_data[i] = {'coords': (x_pixel, y_pixel), 'confidence': point.confidence()}

                lh_idx = keypoints_map['left_hip']
                ls_idx = keypoints_map['left_shoulder']
                string_to_print += "    Left Torso Analysis:\n"
                if lh_idx in points_data and ls_idx in points_data:
                    point_hip_data = points_data[lh_idx]
                    point_shoulder_data = points_data[ls_idx]

                    # Check confidence scores
                    if point_hip_data['confidence'] > KEYPOINT_CONFIDENCE_THRESHOLD and \
                       point_shoulder_data['confidence'] > KEYPOINT_CONFIDENCE_THRESHOLD:
                        point_hip = point_hip_data['coords']
                        point_shoulder = point_shoulder_data['coords']
                        string_to_print += f"      L-Hip: {point_hip} (Conf: {point_hip_data['confidence']:.2f}), L-Shoulder: {point_shoulder} (Conf: {point_shoulder_data['confidence']:.2f})\n"

                        dx = point_shoulder[0] - point_hip[0]
                        dy = point_shoulder[1] - point_hip[1]
                        effective_angle_from_horizontal_l = 90.0

                        if dx == 0 and dy == 0:
                            string_to_print += "      L-Torso points identical, angle undefined (treated as vertical).\n"
                        else:
                            angle_deg_l = math.degrees(math.atan2(dy, dx))
                            abs_angle_deg_l = abs(angle_deg_l)
                            if abs_angle_deg_l > 90:
                                effective_angle_from_horizontal_l = 180 - abs_angle_deg_l
                            else:
                                effective_angle_from_horizontal_l = abs_angle_deg_l
                            string_to_print += f"      L-Torso Raw Angle: {angle_deg_l:.1f} deg, Effective Horiz. Angle: {effective_angle_from_horizontal_l:.1f} deg\n"

                        if effective_angle_from_horizontal_l < degreethres:
                            is_lying_down = True
                            alert_reasons.append(f"L-Torso({effective_angle_from_horizontal_l:.0f}deg)")
                            string_to_print += "      Lying Trigger: L-Torso Horizontal\n"
                        else:
                            string_to_print += "      L-Torso Horizontal (<thresdeg): No\n"
                    else:
                        string_to_print += f"      L-Hip or L-Shoulder keypoint confidence too low (hip: {point_hip_data['confidence']:.2f}, shoulder: {point_shoulder_data['confidence']:.2f}). Skipping angle calculation.\n"
                else:
                    string_to_print += "      L-Hip or L-Shoulder keypoints missing for angle calculation.\n"

                rh_idx = keypoints_map['right_hip']
                rs_idx = keypoints_map['right_shoulder']
                string_to_print += "    Right Torso Analysis:\n"
                if rh_idx in points_data and rs_idx in points_data:
                    point_hip_data = points_data[rh_idx]
                    point_shoulder_data = points_data[rs_idx]

                    # Check confidence scores
                    if point_hip_data['confidence'] > KEYPOINT_CONFIDENCE_THRESHOLD and \
                       point_shoulder_data['confidence'] > KEYPOINT_CONFIDENCE_THRESHOLD:
                        point_hip = point_hip_data['coords']
                        point_shoulder = point_shoulder_data['coords']
                        string_to_print += f"      R-Hip: {point_hip} (Conf: {point_hip_data['confidence']:.2f}), R-Shoulder: {point_shoulder} (Conf: {point_shoulder_data['confidence']:.2f})\n"

                        dx = point_shoulder[0] - point_hip[0]
                        dy = point_shoulder[1] - point_hip[1]
                        effective_angle_from_horizontal_r = 90.0

                        if dx == 0 and dy == 0:
                            string_to_print += "      R-Torso points identical, angle undefined (treated as vertical).\n"
                        else:
                            angle_deg_r = math.degrees(math.atan2(dy, dx))
                            abs_angle_deg_r = abs(angle_deg_r)
                            if abs_angle_deg_r > 90:
                                effective_angle_from_horizontal_r = 180 - abs_angle_deg_r
                            else:
                                effective_angle_from_horizontal_r = abs_angle_deg_r
                            string_to_print += f"      R-Torso Raw Angle: {angle_deg_r:.1f} deg, Effective Horiz. Angle: {effective_angle_from_horizontal_r:.1f} deg\n"

                        if effective_angle_from_horizontal_r < degreethres:
                            is_lying_down = True
                            alert_reasons.append(f"R-Torso({effective_angle_from_horizontal_r:.0f}deg)")
                            string_to_print += "      Lying Trigger: R-Torso Horizontal\n"
                        else:
                            string_to_print += "      R-Torso Horizontal (<thresdeg): No\n"
                    else:
                        string_to_print += f"      R-Hip or R-Shoulder keypoint confidence too low (hip: {point_hip_data['confidence']:.2f}, shoulder: {point_shoulder_data['confidence']:.2f}). Skipping angle calculation.\n"
                else:
                    string_to_print += "      R-Hip or R-Shoulder keypoints missing for angle calculation.\n"

                if user_data.use_frame and frame is not None:
                    for name, index in keypoints_map.items():
                        if index in points_data and points_data[index]['confidence'] > KEYPOINT_CONFIDENCE_THRESHOLD:
                            x, y = points_data[index]['coords']
                            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                        elif index in points_data and points_data[index]['confidence'] <= KEYPOINT_CONFIDENCE_THRESHOLD:
                            # Optionally draw low-confidence keypoints differently, or not at all
                            x, y = points_data[index]['coords']
                            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1) # Red for low confidence
            else:
                string_to_print += "    Landmarks: No landmarks found for this person detection.\n"

            string_to_print += f"    Overall Lying Down Status: {is_lying_down}\n"

            if is_lying_down:
                string_to_print += f"      Final Alert Reasons: {', '.join(alert_reasons)}\n"

                # Calculate center of the bounding box
                bbox_center_x = int((bbox.xmin() + bbox.width() / 2) * width)
                bbox_center_y = int((bbox.ymin() + bbox.height() / 2) * height)
                string_to_print += f"    BBox Center: ({bbox_center_x}, {bbox_center_y})\n"

                # Check if bounding box center is within the horizontal frame center range
                is_horizontally_centered = False
                if left_bound <= bbox_center_x <= right_bound:
                    string_to_print += f"    BBox center X ({bbox_center_x}) is within frame center range ({left_bound:.0f} to {right_bound:.0f}).\n"
                    is_horizontally_centered = True
                else:
                    string_to_print += f"    BBox center X ({bbox_center_x}) is NOT within frame center range ({left_bound:.0f} to {right_bound:.0f}).\n"

                # Check if bounding box center is within the lower 7/10 of the screen height
                is_in_lower_3_5 = False
                lower_3_5_threshold = height * (3/10)  # The upper 3/10 is the area *above* the lower 7/10
                if bbox_center_y >= lower_3_5_threshold:
                    string_to_print += f"    BBox center Y ({bbox_center_y}) is in lower 7/10 of screen (>{lower_3_5_threshold:.0f}).\n"
                    is_in_lower_3_5 = True
                else:
                    string_to_print += f"    BBox center Y ({bbox_center_y}) is NOT in lower 7/10 of screen (>{lower_3_5_threshold:.0f}).\n"

                # Combine all conditions for fall detection
                current_fall_condition_met = is_horizontally_centered and is_in_lower_3_5

            if is_lying_down and user_data.use_frame and frame is not None:
                alert_text = "Alert: Lying Down (" + ", ".join(alert_reasons) + ")"
                alert_x = int(bbox.xmin() * width)
                alert_y = int(bbox.ymin() * height) - 10
                if alert_y < 10: alert_y = int(bbox.ymax() * height) + 20
                cv2.putText(frame, alert_text, (alert_x, alert_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            string_to_print += "  Detected Keypoints (Pixel Coords, Confidence):\n"
            if len(points_data) > 0:
                 for name, index in keypoints_map.items():
                    if index in points_data:
                        x, y = points_data[index]['coords']
                        conf = points_data[index]['confidence']
                        string_to_print += f"    {name}: ({x}, {y}) Conf: {conf:.2f}\n"
            else:
                string_to_print += "    No keypoints processed or available to list.\n"
            string_to_print += "---\n"

    # --- Timer Logic for Fall Detection ---
    if current_fall_condition_met:
        if not user_data.fall_detection_initiated:
            # First frame where fall condition (lying + centered) is met
            user_data.fall_detection_initiated = True
            user_data.fall_start_time = datetime.now()
            string_to_print += f"\n>>> Fall detected and centered. Starting {FALL_TIMER_SECONDS} second confirmation timer...\n"
            if user_data.motor_controller:
                # Stop the motors immediately when fall condition is met
                user_data.motor_controller.stop_all()
        else:
            # Fall condition continues to be met, check elapsed time
            elapsed_time = (datetime.now() - user_data.fall_start_time).total_seconds()
            remaining_time = max(0, FALL_TIMER_SECONDS - elapsed_time)
            string_to_print += f"\n>>> Fall detected and centered. Timer active: {elapsed_time:.1f}/{FALL_TIMER_SECONDS}s elapsed ({remaining_time:.1f}s remaining).\n"

            if elapsed_time >= FALL_TIMER_SECONDS:
                # Timer elapsed, CONFIRM fall, log and exit
                if not user_data.fall_detected_and_stopping: # Ensure we only do this once
                    string_to_print += "\n!!! CONFIRMED FALL DETECTED (Timer Elapsed). SCHEDULING SCRIPT EXIT. !!!\n"
                    # Log the coordinates to file (using the last detected person's bbox_center_x, bbox_center_y, track_id)
                    log_fall_coordinates(bbox_center_x, bbox_center_y, track_id)
                    user_data.fall_detected_and_stopping = True
                    # Motors should already be stopped by the initial fall detection.
                    GLib.idle_add(attempt_app_shutdown_async, user_data.app_instance)
            # Else, timer is still running, do nothing until it elapses.
    else:
        # Fall condition (lying down AND centered) is NOT met in this frame
        if user_data.fall_detection_initiated:
            # If the timer was active, reset it because the condition is broken
            string_to_print += "\n>>> Fall condition (lying down AND centered) no longer met. Resetting timer.\n"
            user_data.fall_detection_initiated = False
            user_data.fall_start_time = None
            # Resume motor here because the fall was not confirmed or person got up
            if user_data.motor_controller:
                # Check if either motor is currently running. If not, resume forward movement.
                if not user_data.motor_controller._running_left.is_set() and \
                   not user_data.motor_controller._running_right.is_set():
                    print(">>> Fall timed out/person got up. Resuming motors forward.")
                    user_data.motor_controller.start(left_direction=1, right_direction=-1)
    # --- End Timer Logic ---


    if user_data.use_frame and frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)

    # If a fall has been detected and we are stopping, remove the probe
    if user_data.fall_detected_and_stopping:
        return Gst.PadProbeReturn.REMOVE

    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# Helper function to attempt app shutdown asynchronously
# -----------------------------------------------------------------------------------------------
def attempt_app_shutdown_async(app_instance):
    """
    Called asynchronously from GLib's idle loop to attempt to shut down the app.
    It first tries to call app_instance.stop(), then falls back to forceful exit.
    """
    print("Executing asynchronous app shutdown attempt...")

    # 1. Try to call a .stop() method on the app_instance itself
    if hasattr(app_instance, 'stop') and callable(app_instance.stop):
        print("Attempting to call app_instance.stop() for graceful shutdown.")
        try:
            app_instance.stop()
            print("app_instance.stop() called. Waiting for app to exit.")
            # We assume app.stop() will handle GStreamer NULL state and GLib loop quit
        except Exception as e:
            print(f"Error calling app_instance.stop(): {e}. Resorting to forceful exit.", file=sys.stderr)
            sys.exit(0) # Forceful exit if app.stop() fails
    else:
        # 2. If no .stop() method, try to set pipeline to NULL (from previous attempts)
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
                sys.exit(0) # Forceful exit as graceful GLib.MainLoop.quit() is not working
            except Exception as e:
                print(f"Error setting pipeline state to NULL: {e}. Resorting to forceful exit.", file=sys.stderr)
                sys.exit(0)
        else:
            print("Warning: Could not find pipeline object in GStreamerPoseEstimationApp instance. Resorting to forceful exit.", file=sys.stderr)
            sys.exit(0) # Forceful exit if no pipeline found

    return GLib.SOURCE_REMOVE # Remove this idle source after execution

# -----------------------------------------------------------------------------------------------
# Helper function for COCO keypoint mapping
# -----------------------------------------------------------------------------------------------
def get_keypoints():
    keypoints = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16,
    }
    return keypoints

# -----------------------------------------------------------------------------------------------
# Function to log fall coordinates to a text file
# -----------------------------------------------------------------------------------------------
def log_fall_coordinates(x_coord, y_coord, person_id):
    """
    Writes the center coordinates of the fallen person's bounding box to a log file.
    """
    # Ensure the log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - Person ID: {person_id}, Fallen at X: {x_coord}, Y: {y_coord}\n"

    try:
        with open(LOG_FILE, "a") as f: # Open in append mode
            f.write(log_entry)
        print(f"Logged fall coordinates to {LOG_FILE}")
    except IOError as e:
        print(f"Error writing to log file {LOG_FILE}: {e}")

# -----------------------------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Pose Estimation App with Detailed Lying Down Detection Logging.")
    print(f"Fall confirmation timer set to {FALL_TIMER_SECONDS} seconds.")
    print("Note: Ensure GStreamerPoseEstimationApp is instantiated with necessary arguments for your Hailo setup.")

    user_data = user_app_callback_class()

    # Initialize the DualStepperMotorController with pins for both motors.
    # IMPORTANT: Adjust these pin numbers to match your actual wiring for Left and Right motors.
    motor_controller = DualStepperMotorController(
        left_in1_pin=17, left_in2_pin=27, left_in3_pin=22, left_in4_pin=24, # Example pins for Left Motor
        right_in1_pin=5, right_in2_pin=6, right_in3_pin=13, right_in4_pin=19, # Example pins for Right Motor
        min_delay=0.0015
    )
    user_data.motor_controller = motor_controller # Pass the controller to user_data


    motor_controller.start(left_direction=1, right_direction=-1) # Start rotation movement

    app = GStreamerPoseEstimationApp(app_callback, user_data)
    user_data.app_instance = app

    try:
        app.run()
        print("app.run() has returned. This indicates the GStreamer pipeline's main loop has quit.")

    except KeyboardInterrupt:
        print("Ctrl+C pressed. Shutting down pipeline...")
    finally:
        # Ensure the pipeline is stopped in case of Ctrl+C if it wasn't by fall detection
        pipeline = None
        if user_data.app_instance:
            if hasattr(user_data.app_instance, '_GstApp__pipeline'):
                pipeline = getattr(user_data.app_instance, '_GstApp__pipeline')
            elif hasattr(user_data.app_instance, 'pipeline'):
                pipeline = getattr(user_data.app_instance, 'pipeline')

        if pipeline and pipeline.get_state(Gst.State.NULL)[1] != Gst.State.NULL:
            print("Ensuring pipeline is set to NULL during final cleanup (if not already).")
            pipeline.set_state(Gst.State.NULL)

        # Clean up the motor controller and gpiozero objects
        motor_controller.cleanup()
        print("Script finished.")