
# -------------------------------------------------------------------------------------------------
################### Importing The Libraries ###################

import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import math
from PIL import Image as PILImage
from PIL import ImageDraw

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.graphics.texture import Texture

# Importing the Dqn object from our AI in ai.py
from final_t3d_new_pil import Train_TD3
# -------------------------------------------------------------------------------------------------

################### Define Variables and other Global Settings ###################

brain = Train_TD3('test_wandering')  # Initialize TD3

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Global Variables
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")
total_reward = 0
episode_done = False
first_update = True
max_action_value = 1
count = 0
car_org_img = PILImage.open("./images/car.png").convert('RGBA')
car_org_img = car_org_img.resize((20, 10))
total_reward_end = -5000
last_distance = 0
loop_timestep_cnt = 20000
last_x = 0
last_y = 0
sand_img_pil = PILImage.open("./images/test_1.png").convert('L')

img = PILImage.open("./images/mask.png").convert('L')
longueur = img.size[0]
largeur = img.size[1]
sand = np.zeros((longueur, largeur))
sand = np.asarray(img)/255
goal_x = 1420
goal_y = 622
first_update = False
swap = 0
timesteps = 0

# -------------------------------------------------------------------------------------------------
################### Helper function to crop image ###################


def get_input_image(x, y, angle):
    crop_size = 60
    x, y, angle = int(x), int(y), int(-angle)

    car_rotated = car_org_img.rotate(angle, expand=1)
    sand_img_copy = sand_img_pil.copy()
    sand_img_copy.paste(car_rotated, (x, y), car_rotated)

    img_patch = sand_img_copy.crop(
        (x, y-crop_size/2, x+crop_size, y+crop_size/2))

    # FOR DEBUGGING
    # img_patch.save('./images_test/car_{}.png'.format(count))
    # global count
    # count += 1
    return img_patch

####################################
# NOT USED
# Crop image with Triangle as car
####################################

# sand_img_pil = PILImage.open("./images/mask.png").convert('RGB')
# def get_input_image(x, y, angle):
#     global count
#     crop_size = 60
#     x, y, angle = int(y), int(x), int(angle-90)

#     base = 15
#     theta = angle * math.pi / 180

#     x1, y1 = x + (4/3) * base * math.cos(theta), y - \
#         (4/3) * base * math.sin(theta)
#     x2, y2 = x + (2/3) * base * math.cos(theta + (135*math.pi/180)
#                                          ), y - (2/3) * base * math.sin(theta + (135*math.pi/180))
#     x3, y3 = x + (2/3) * base * math.cos(theta + (225*math.pi/180)
#                                          ), y - (2/3) * base * math.sin(theta + (225*math.pi/180))

#     sand_img_copy = sand_img_pil.copy()
#     draw = ImageDraw.Draw(sand_img_copy)
#     draw.polygon([(x1, y1), (x2, y2), (x, y),
#                   (x3, y3), (x1, y1)], fill=(255, 0, 0))

#     img_patch = sand_img_copy.crop(
#         (x, y-crop_size/2, x+crop_size, y+crop_size/2))

#     # img_patch.resize((60, 60)).convert('L').save(
#     #     "./images_test/image_" + str(count) + ".png")
#     count = count + 1

#     return img_patch.convert('L')

# -------------------------------------------------------------------------------------------------
################### Car widget ###################


class Car(Widget):

    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation

# -------------------------------------------------------------------------------------------------
################### Game widget ###################


class Game(Widget):

    car = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):
        # Global variables
        global sand
        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        global total_reward
        global episode_done
        global timesteps
        global total_reward_end

        # Width and height of the entire map 1429x660
        longueur = self.width
        largeur = self.height

        # distance in x and y from the goal
        # using this cordinate we can calculate the angle to move towards the target
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y

        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.
        curret_img = get_input_image(self.car.x, self.car.y, self.car.angle)
        norm_last_distance = last_distance / int(1574)

        last_signal = [[curret_img, norm_last_distance,
                        orientation, -orientation], episode_done]

        # Output from the network
        action = max_action_value * brain.update(last_reward, last_signal)

        # action = np.random.randint(-3, 3)
        rotation = int(action)
        self.car.move(rotation)

        # the distance for final goal
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)

        # Reward Function
        distance_reward = 1 - norm_last_distance
        road_reward = 1 - norm_last_distance

        if timesteps < 2 * loop_timestep_cnt:
            if distance < last_distance:
                last_reward = distance_reward
                self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            else:
                last_reward = -1
                self.car.velocity = Vector(.5, 0).rotate(self.car.angle)

        elif timesteps >= 2 * loop_timestep_cnt and timesteps < 8*loop_timestep_cnt:
            if sand[int(self.car.x), int(self.car.y)] > 0:
                self.car.velocity = Vector(.5, 0).rotate(self.car.angle)
                last_reward = -1
            else:
                self.car.velocity = Vector(1, 0).rotate(self.car.angle)
                last_reward = road_reward

        else:
            if sand[int(self.car.x), int(self.car.y)] > 0:
                self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
                last_reward = -1
            else:
                self.car.velocity = Vector(1, 0).rotate(self.car.angle)
                if distance < last_distance:
                    last_reward = .1
                else:
                    last_reward = - .2

        # reset car if the car is near the wall
        distance_from_wall_x = 10
        # vel_discount = 1 - max(self.car.velocity, .1) ^ (1/max(distance), .1)
        if self.car.x < distance_from_wall_x:
            self.car.x = distance_from_wall_x
            last_reward = -1
        if self.car.x > self.width - distance_from_wall_x:
            self.car.x = self.width - distance_from_wall_x
            last_reward = -1
        if self.car.y < distance_from_wall_x:
            self.car.y = distance_from_wall_x
            last_reward = -1
        if self.car.y > self.height - distance_from_wall_x:
            self.car.y = self.height - distance_from_wall_x
            last_reward = -1

        episode_done = False
        total_reward += last_reward

        # check if the car has reached the goal and swap
        if total_reward < total_reward_end:
            print("----------Resetting Episode----------")
            episode_done = True
            total_reward = 0
            self.car.x = 9  # int(np.random.randint(25, self.width-25, 1)[0])
            self.car.y = 85  # int(np.random.randint(25, self.height-25, 1)[0])
            self.car.angle = np.random.randint(0, 360)
            print("----------new car location----------", self.car.x, self.car.y)

        if distance < 25:
            print("----------reached destination----------")
            if timesteps < 10*loop_timestep_cnt:
                if swap == 1:

                    episode_done = True
                    total_reward = 0
                    goal_x = 1420
                    goal_y = 622
                    swap = 0
                else:
                    goal_x = 9
                    goal_y = 85
                    swap = 1
                print("----------new goals are----------", goal_x, goal_y)
            else:
                if swap == 1:
                    episode_done = True
                    total_reward = 0
                    goal_x = 1192
                    goal_y = 444
                    swap = 0
                else:
                    goal_x = 9
                    goal_y = 85
                    swap = 1
                print("----------new goals are----------", goal_x, goal_y)

        last_distance = distance
        timesteps += 1

# -------------------------------------------------------------------------------------------------
################### Running the whole thing ###################


class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        return parent


if __name__ == '__main__':
    CarApp().run()

# -------------------------------------------------------------------------------------------------
################### THE END ###################
