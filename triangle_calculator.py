# BEN MARSH - F5037922
# AI HAS NOT BEEN USED TO ANY DEGREE WITHIN THIS WORK

import math

import sys

#title

print("\nTRIANGLE CALCULATOR\n")

#ensuring proper amount of command line arguments

if len(sys.argv)!=3:
    sys.exit("Input error to many input arguments given\n")

#gathering data from user for base and height

height = sys.argv[1]

base = sys.argv[2]

# converting inputted data to integers for mathematical use and checking for invalid data types 

try:
    height = int(height)
except ValueError as e:
    print("Data type not integers please use integers PROGRAM WILL TERMINATE")
    sys.exit(e)
except Exception as e:
    print("General Error occurred please try again, PROGRAM WILL TERMINATE")
    sys.exit(e)
    
try:
    base = int(base)
except ValueError as e:
    print("Data type not integers please use integers PROGRAM WILL TERMINATE")
    sys.exit(e)
except Exception as e:
    print("General Error occurred please try again, PROGRAM WILL TERMINATE")
    sys.exit(e)
    

#making sure inputted values are in acceptable range between 1 and 10

if height > 10:
    sys.exit("Inputted value outside acceptable range")   
elif height < 1:
    sys.exit("Inputted value outside acceptable range") 

if base > 10:
    sys.exit("Inputted value outside acceptable range")   
elif base < 1:
    sys.exit("Inputted value outside acceptable range") 

#calculating the three desired answers

#hypothenuse of triangle

#formula for calculation using pythagoras 

#www.w3schools.com. (n.d.). Python math.sqrt() Method. [online] Available at: https://www.w3schools.com/python/ref_math_sqrt.asp [Accessed 10 Oct. 2025].

#The math.sqrt function returns the squrae root of a number or variable 

hypothenus = math.sqrt((base*base) + (height*height))

#printing necessary information 

print("\nHYPOTHENUS")

print("\nThe hypothenus rounded to decimals is",round(hypothenus,2))

#Angle alpha

#declaring variable

angle_alpha = None

#finding radian amount of inputted values

#W3schools.com. (2025). Welcome To Zscaler Directory Authentication. [online] Available at: https://www.w3schools.com/python/ref_math_atan2.asp [Accessed 10 Oct. 2025].

#The math.atan2 function returns the arc tangent of two variables or values in the measurement radians

angle_alpha_radian = (math.atan2(height, base))

#converting inputted radians to degrees

#W3schools.com. (2025a). W3Schools.com. [online] Available at: https://www.w3schools.com/python/ref_math_degrees.asp [Accessed 10 Oct. 2025].

#The math.degrees function is used in this code to convert the output of the variable angle_alpha_radian to degrees

angle_alpha_degrees = (math.degrees(angle_alpha_radian))

#switching variable to clearer name for ease of use

angle_alpha = angle_alpha_degrees

#printing necessary information

print("\nANGLE ALPHA")

print("\nThe size of angle alpha rounded to 2 decimals is:",round(angle_alpha, 2),"degrees")

#Angle beta

angle_beta = None

#declaring the right  angle 

right_angle = int(90)

#doing calculation to find missing angle

missing_angle = 180 - (right_angle + angle_alpha)

#converting angle_beta to float

angle_beta = float(missing_angle)

#displaying the results neatly

print("\nANGLE BETA")

print("\nThe size of angle beta rounded to 2 decimals is:",round(angle_beta, 2),"degrees")
