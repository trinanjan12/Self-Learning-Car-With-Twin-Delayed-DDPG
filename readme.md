## Self Learning Car With Reinforcement Learning 

## Update 
    # Work in Progress as of May 10 2020
    1. working with continuous function reward strategy(no more absolute values). 
    This makes the learning better
    source https://medium.com/@BonsaiAI/deep-reinforcement-learning-models-tips-tricks-for-writing-reward-functions-a84fe525e8e0
    2. currently the car runs between 2 points easily, 
    but the moment i start giving reward while the car is on sand 
    i can see that the car tires to be on the road, but somehow distance is overpowering, so it goes out of road.
    3. Thinking about how to give reward for being on road, 
    it should not be only be based on distance. thinking about  
    getting distance from the middle of the track and add it as state parameter.
    Only problem is for this i need to draw a map such that i can query the middle point.
    
    
    

### Project Statement :
Making a reinforcement learning agent(car) to travel around a city map and to reach from point A to Point B while taking the road and avoiding sand 


### Environment Description :
Map : I have taken a city map. The map has roads and buildings. The roads are where the rl agent will learn to walk/drive. 
Mask : We take the same city map black and white mask. Road is black and sand is white.
Sand : sand is basically 90 deg (anti clock wise) rotated map of Mask. we will be using this sand map for training.

### Agent Description :
1. Car : Car image is used as rl agent
2. Action dimension : 1 (rotation)
3. State dimension : 4 (image crop of the current location of size (100,100), distance between the agent and the goal, orientation, - - orientation)
4. State dimension description :
    1. Image crop: Crop image of the car on the sand to tell the n/w that it is on sand or road
    2. Distance : euclidean distance between the agent and the goal point
    3. Orientation: Angle between the car axis and the goal 

### Image crops:
<p align='center'><img src="https://i.imgur.com/qWX5XRQ.png"></p>

### Algorithm Used :
I have used Twin delayed DDPG Algorithm. I will add description about the algorithm later

### Rewards Strategy:
This is the final strategy (i have added other things that i have tried in the bottom section)

1. First train the model on the distance and without sand
2. Once the model is mature enough and the rewards starts cumulating train the network on sand and without any destination
3. Once both is trained add sand and destination. Also keep a living penalty to minimize the time

### Episode Ending Strategy :
1. Episode is ended based on which part of the reward strategy is being done(described above)
2. It gradually increases for the above steps (-4000, - 8000, -10000) 
3. These negative values are cumulative gradients and are used to update done variable

### TD3 Architecture:
    I have used the same TD3 architecture and the image is encoded with the below convolution blocks

    ##############################
    # Total params: 6,272
    # Trainable params: 6,272
    # Non-trainable params: 0
    ##############################

    def ImageConv(in_dim, out_dim):
        model = nn.Sequential(
            nn.Conv2d(in_dim, 32, kernel_size=3,
                    stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(.2),

            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(.2),

            nn.Conv2d(32, 16, kernel_size=1, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(9),
            nn.Flatten(),
            nn.Linear(16, out_dim, bias=False))
        return model





## Output Videos 

[![Trained only for Distance](https://youtu.be/TlAh78Tc1HI)](https://youtu.be/TlAh78Tc1HI)

[![Trained only for Sand](https://youtu.be/TlAh78Tc1HI)](https://www.youtube.com/watch?v=Afhnsyr_oRo)

#### Trained Sand + Distance:  
Incoming

## Things Done 

1. Integrated TD3 with convolution
2. Croped image of car on sand with angle 
3. Training is happening partially for 3 different reward strategy (Rewards Strategy section)
     

### Challanges Faced and Possible solution

1. kivy and pillow/numpy has different co-ordinate system
2. Loading a pillow image and converting it to numpy tranposes the image. Because of  images were going wrong to the network
3. Initially when i tried the network directly without following the reward strategy described above the car starts rotating. I was thinking because of numpy and pillow conversion images were wrong
4. Later when i started following the reward strategy(step by step training) the rotation issue was fixed
5. later realized the entire network trains based on the reward strategy and nothing much change is required on TD3
6. I have tried different reward strategy and reward values
7. Things i tried overall to stop rotation
    - used pretrained network embeddings(trained on the cropped images on BCE Loss sand vs road)
    - passed different orientation images 
    - passed images with and without other state variables like distance and orientation
    - tried changing max action values and temperature parameter change to stop the rotation issue 
    - also tried gradient clipping to stop rotation 
        

### ToDo:
1. Need to plot the graph for sand/distance accumulated rewards to design the optimum training rewards
2. Final training to keep the car on road

