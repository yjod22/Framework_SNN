#!/usr/bin/env python
# ~/Documents/tepic/Git/emojiMe/workspace/src/emojime/src
import os
from PIL import Image, ImageDraw, ImageFont

# show image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import random

import numpy as np
import cv2

import threading
import time
import sys

global satisfied
global shocked
global neutral
global bored
global stopFlag
	
satisfied = 55
shocked   = 15
neutral   = 15
bored     = 15
stopFlag = False

# text_color
yellow_text = (255, 255, 0)
black_text = (0,0,0)

# background
green = (34,139,34)
yellow = (255,255,0)
orange = (255,165,0)
red = (255,0,0)
black = (0,0,0)

y_offset = -65
wait_1s = 1000

def getColor(percentages):
	satisfied = percentages[0]

	ordered = [percentages[0], percentages[1], percentages[2], percentages[3]]

	ordered.sort()
	value_1 = ordered[3] # MAX
	value_2 = ordered[2]
	value_3 = ordered[1]
	value_4 = ordered[0] # MIN

	if(value_1==satisfied):
		if(value_1>=2*value_2):
			return green
		if(value_1>=1.5*value_2):
			return yellow
		if(value_1>value_2):
			return orange
		else:
			return red
	else:
		return red

def createResults_time(percentages,minutes,seconds):
	createResults(percentages)	
	canvas = Image.open('Result_Image.png')
	background= ImageDraw.Draw(canvas)
	clock_font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationMono-BoldItalic.ttf', 64)
	if(seconds<10):
		if(minutes<10):
			clock_text = '0'+str(minutes)+':0'+str(seconds)
		else:
			clock_text = str(minutes)+':0'+str(seconds)
	else:
		if(minutes<10):
			clock_text = '0'+str(minutes)+':'+str(seconds)
		else:
			clock_text = str(minutes)+':'+str(seconds)

	clock_location = (800,680)
	background.text(clock_location, clock_text, font=clock_font, fill=black_text)
	canvas.save('Result_Image.png')
	canvas.close()

def createResults(percentages):
	resize_coeff_satisfied = percentages[0]
	resize_coeff_shocked   = percentages[1]
	resize_coeff_neutral   = percentages[2]
	resize_coeff_bored     = percentages[3]

	background_color = getColor(percentages)
	canvas = Image.new('RGB', (1024, 768), color = background_color)
	background= ImageDraw.Draw(canvas)

	title_font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationMono-BoldItalic.ttf', 36)
	title_text = "emojiME - audiance response"
	title_location = (240,40)
	background.text(title_location, title_text, font=title_font, fill=black_text)

	emoji_font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationMono-BoldItalic.ttf', 48)

	resize_coeff_satisfied = resize_coeff_satisfied*2.0/100
	satisfied_png = Image.open('./emojies/satisfied.png', 'r')
	dimensions = satisfied_png.size
	new_dimensions = (int(math.floor(resize_coeff_satisfied*dimensions[0])), int(math.floor(resize_coeff_satisfied*dimensions[1])))
	if(new_dimensions[0]<30):
		new_dimensions = (30,30)
	satisfied_png = satisfied_png.resize(new_dimensions, Image.ANTIALIAS)
	satisfied_png_location = (160-int(math.floor(new_dimensions[0]/2)), 400-int(math.floor(new_dimensions[1]/2))+y_offset)
	satisfied_location = (128-18-30,502+y_offset) #256-18
	satisfied_text = str(resize_coeff_satisfied*50.0)+'%' # it was scalled x2, up to 200%, and then divided by 100

	resize_coeff_shocked = resize_coeff_shocked*2.0/100
	shocked_png = Image.open('./emojies/shocked.png', 'r')
	dimensions = shocked_png.size
	new_dimensions = (int(math.floor(resize_coeff_shocked*dimensions[0])), int(math.floor(resize_coeff_shocked*dimensions[1])))
	if(new_dimensions[0]<30):
		new_dimensions = (30,30)
	shocked_png = shocked_png.resize(new_dimensions, Image.ANTIALIAS)
	shocked_png_location = (225+160-int(math.floor(new_dimensions[0]/2)), 400-int(math.floor(new_dimensions[1]/2))+y_offset)
	shocked_location = (384-2*18-30,502+y_offset) #384-36
	shocked_text = str(resize_coeff_shocked*50.0)+'%'

	resize_coeff_neutral = resize_coeff_neutral*2.0/100
	neutral_png = Image.open('./emojies/neutral.png', 'r')
	dimensions = neutral_png.size
	new_dimensions = (int(math.floor(resize_coeff_neutral*dimensions[0])), int(math.floor(resize_coeff_neutral*dimensions[1])))
	if(new_dimensions[0]<30):
		new_dimensions = (30,30)
	neutral_png = neutral_png.resize(new_dimensions, Image.ANTIALIAS)
	neutral_png_location = (2*235+160-int(math.floor(new_dimensions[0]/2)), 400-int(math.floor(new_dimensions[1]/2))+y_offset)
	neutral_location = (640-3*18-30,502+y_offset) #640-54
	neutral_text = str(resize_coeff_neutral*50.0)+'%'

	resize_coeff_bored = resize_coeff_bored*2.0/100
	bored_png = Image.open('./emojies/bored.png', 'r')
	dimensions = bored_png.size
	new_dimensions = (int(math.floor(resize_coeff_bored*dimensions[0])), int(math.floor(resize_coeff_bored*dimensions[1])))
	if(new_dimensions[0]<30):
		new_dimensions = (30,30)
	bored_png = bored_png.resize(new_dimensions, Image.ANTIALIAS)
	bored_png_location = (3*235+160-int(math.floor(new_dimensions[0]/2)), 400-int(math.floor(new_dimensions[1]/2))+y_offset)
	bored_location = (896-4*18-30,502+y_offset) #896-72
	bored_text = str(resize_coeff_bored*50.0)+'%'

	canvas.paste(satisfied_png, satisfied_png_location,satisfied_png)
	background.text(satisfied_location, satisfied_text, font=emoji_font, fill=black_text)

	canvas.paste(shocked_png, shocked_png_location,shocked_png)
	background.text(shocked_location, shocked_text, font=emoji_font, fill=black_text)

	canvas.paste(neutral_png, neutral_png_location,neutral_png)
	background.text(neutral_location, neutral_text, font=emoji_font, fill=black_text)

	canvas.paste(bored_png, bored_png_location,bored_png)
	background.text(bored_location, bored_text, font=emoji_font, fill=black_text)
	 
	canvas.save('Result_Image.png')
	canvas.close()
	#return canvas

def getNewResults():
	global satisfied
	global shocked
	global neutral
	global bored
	global stopFlag
	global percentages
    
	while(True):
		# input = raw_input("Do you want to terminate [yes = y]? ")
		# if(input=='y' or input=='yes'):
		# 	stopFlag = True
		# 	#break
		# else:
		# 	stopFlag = False
		input = raw_input("satisfied, shocked, neutral, bored: ")
		input     = input.split(",")
		#satisfied = float(raw_input("Satisfied: "))
		# shocked   = float(raw_input("Shocked: "))
		# neutral   = float(raw_input("Neutral: "))
		# bored     = float(raw_input("Bored: "))
		if(len(input) == 4):
			satisfied = float(input[0])
			shocked   = float(input[1])
			neutral   = float(input[2])
			bored     = float(input[3])
			setNewResults(satisfied,shocked,neutral,bored)#,stopFlag)
			#percentages = [satisfied,shocked,neutral,bored]

def getNewResults_evaluation():
	global satisfied
	global shocked
	global neutral
	global bored
	global stopFlag
	global percentages

	# elements = [100, 0, 0, 0, 80, 0, 15, 5, 40, 15, 25, 20, 30, 15, 30, 25, 60, 5, 15, 20, 65, 5, 15, 15]
	elements = [55, 15, 15, 25]

	round = 0
	while(True):
		print("Press 1..4 to change emotion response:")
		print("1 - satisfied (green)")
		print("2 - fine, but people getting bored")
		print("3 - people almost sleep")
		print("4 - everybody is dead")
		input = raw_input(">> ")

		if(input!=''):
			input = int(input)

		if(input==1):
			elements[0] = random.randrange(60,70,1) # 2+
			elements[2] = random.randrange(10,20,1)
			elements[3] = random.randrange(5,10,1)
			elements[1] = 100-(elements[0]+elements[2]+elements[3])
		else:
			if(input==2):
				elements[0] = random.randrange(45,49,1) # 1,5..2
				elements[2] = random.randrange(25,30,1)
				elements[3] = random.randrange(15,20,1)
				elements[1] = 100-(elements[0]+elements[2]+elements[3])
			else:
				if(input==3):
					elements[0] = random.randrange(30,37,1) # 1..1,5
					elements[2] = random.randrange(25,29,1)
					elements[3] = random.randrange(25,29,1)
					elements[1] = 100-(elements[0]+elements[2]+elements[3])
				else:
					if(input==4):
						elements[0] = random.randrange(20,25,1) # 1-
						elements[2] = random.randrange(25,30,1)
						elements[3] = random.randrange(25,30,1)
						elements[1] = 100-(elements[0]+elements[2]+elements[3])									

		satisfied = elements[round*4+0]
		shocked = elements[round*4+1]
		neutral = elements[round*4+2]
		bored = elements[round*4+3]
		setNewResults(satisfied,shocked,neutral,bored)

def setNewResults(satisfied_input,shocked_input,neutral_input,bored_input):#,status=False):
	global satisfied
	global shocked
	global neutral
	global bored
	#global stopFlag
	global percentages
    
	satisfied = satisfied_input
	shocked   = shocked_input
	neutral   = neutral_input
	bored     = bored_input
	#stopFlag  = status
	percentages = [satisfied,shocked,neutral,bored]
	#print(stopFlag)
	
percentages = [satisfied,shocked,neutral,bored]
# createResults(percentages)



# display_thread = threading.Thread(target=setNewResults)
# display_thread.start()

def main():	
	pressed = 0
	minutes = 0
	seconds = 0

	while(not stopFlag):
		createResults_time(percentages,minutes,seconds)
		output = cv2.imread("Result_Image.png")
		cv2.imshow('Results',output)
		cv2.waitKey(wait_1s)
		seconds = seconds+1
		if(seconds==60):
			seconds = 0
			minutes = minutes+1
			
	print("Shutting down")

if __name__ == '__main__':

	useROS = False
	try:
		if(useROS):			
			input_thread = threading.Thread(target=getNewResults)
		else:
			input_thread = threading.Thread(target=getNewResults_evaluation)
		
		#enables the thread to kill itself after the end of the program has been reached
		# this line must be executed before the thread has started
		input_thread.daemon = True

		input_thread.start()

		main()
	except (KeyboardInterrupt, SystemExit):
	    sys.exit()