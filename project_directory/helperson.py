from datetime import datetime
import sys
import time
from gpiozero import Button
import smtplib
from email.message import EmailMessage
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    """This function takes text and speaks it out loud."""
    engine.say(text)
    engine.runAndWait()

real_fall_flag = False

# Setup button on GPIO pin 15 (BOARD 10)
button = Button(15, pull_up=True)
countT = 60

def countdown(t):
    """Countdown function that checks for a button press to abort."""
    global real_fall_flag
    while t:
        mins, secs = divmod(t, 60)
        if button.is_pressed:
            speak("System reset by user.")
            sys.exit(0)
        time.sleep(1)
        t -= 1
    real_fall_flag = True

# Initial alert
speak("Fall detected. If you do not need help, hold the reset button.")
print("Fall detected. If you do not need help, hold the reset button.")
countdown(int(countT))

if real_fall_flag:
    speak("Person needs help. Sending an email for assistance.")
    print("Person needs help. Sending an email for assistance.")
    # Email content
    subject = "PERSON NEEDS HELP"
    body = "THIS IS A TEST FOR OUR PROJECT"
    sender_email = "<SENDER>@gmail.com" # it is recommended to open a new gmail account for the robot and generate an api key for this new accout
    receiver_email = "<RECIVER>@gmail.com"
    password = "gmail api key"  # put your gmail api key here 

    # Create the email message
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    # Send the email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, password)
            smtp.send_message(msg)
        speak("Email sent successfully.")
        print("Email sent successfully.")
    except Exception as e:
        error_message = f"Failed to send email: {e}"
        print(error_message) # Print the error for debugging
        speak("Failed to send the email.")
        print("Failed to send the email.")
        sys.exit(1)

    # Wait for button press to reset
    speak("Waiting for a button press to reset the system.")
    print("Waiting for a button press to reset the system.")
    button.wait_for_press()
    speak("Button pressed by helper after fall. Resetting the system now.")
    print("Button pressed by helper after fall. Resetting the system now.")
    sys.exit(0)