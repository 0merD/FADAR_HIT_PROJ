![2fab83da-c91f-4af4-85c7-2cefc2211d6b](https://github.com/user-attachments/assets/6dc1a2e6-1bf3-4375-b92d-afe55065f053)Download only project directory and add it to the root folder of the hailo rpi5 examples repo on your Rpi 5

To activate the project first use the following commands

#change the gmail api key to your api key in  helperson.py
<YOURGMAIAPIKEY>
# change directory to hailo-rpi5-examples
cd <yourpathhere>/hailo-rpi5-examples
# run the virtual environment 
source setup_env.sh
# assuming you have the assembled robot change directory to project directory
cd /project_directory
# run the main loop
python mainloop.py

robotics platform files:
https://www.printables.com/model/1344093-robotics-platform-for-raspberry-pi-5-with-28-byj-4

# assembly  instructions
use 2 l298n DC motor drivers
use 2 28byj-48 stepper motors
hook up the motors to the drivers in the following order
out1 to blue motor wire
out2 to pink motor wire
out3 to yellow motor wire
out4 to orange motor wire

use an otterbox obftc 0041 A powerbank (or smaller) as a powersource for the drivers 
use a raspberry pi 5 with an hailo AI kit or AI HAT+
use a UPS HAT of your choosing to power the logic while disconnected from electricity (original project used geekworm X1200)
use 1 pushbutton
use a light bluetooth or usb speaker of your choosing. we used a jbl go3
use a Pi camera module 3 75 degrees FOV (either regular or without IR filter)
breadbord recommanded but optional
use a striped usb wire or a usb to pin adapter to provide  voltage to the motor drivers 
aqquire 4 8x15x5 ball bearings for the support wheels
aqquire or print 2 silicone/rubber/tpu wrist bands as tires for the driving wheels 
print the robot's body, and wheels (4 thin support wheels and 2 thick driving wheels) out of PETG (we used sanlu elitepetg in space gray) with 10% gyroid infill and 3 walls 
print at least 2 battery plugs if you use otterbox obftc 0041 as a power bank (recommended at both 100 and 105% scale for minor variances)
print the 4 connection lugs that will connect the robot's body to the wheels
install the bearings onto the support wheels.
insert the connection lugs thick side into the robots main body
insert the 2 stepper motors into their housing on the robots main body
connect the driving wheels to the stepper motors
install the tires onto the driver wheels
gently screw your rapsberry pi assembply into the case with the usb side faceing away from the powerbank slot and towards the camera holder
insert the battery bank 
insert the battery holder plugs to hold the battery
connet the pi camera to the Rpi
mount the pi camera on the from of the robot on the dedicated pi camera 3 housing

connect the gpio pins to the motors in the following order and use this delay 
left_in1_pin=17, left_in2_pin=27, left_in3_pin=22, left_in4_pin=24, # Example pins for Left Motor
right_in1_pin=5, right_in2_pin=6, right_in3_pin=13, right_in4_pin=19, # Example pins for Right Motor
min_delay=0.0015
to out 1-4 on the drivers accordingly
connect the pushbutton to gpio 15 and to ground

connect a bluetooth or usb speaker to your raspberry pi

![Screenshot 2025-07-02 104456](https://github.com/user-attachments/assets/5f60053e-9d37-4e85-ac1a-ec5816009ea7)



