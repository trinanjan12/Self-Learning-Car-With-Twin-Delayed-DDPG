import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms


# -------------------------------------------------------------------------------------------------
# Helper function to build the cnn network

##############################
# Total params: 6,272
# Trainable params: 6,272
# Non-trainable params: 0
##############################


def ImageConv(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, 32, kernel_size=3,
                  stride=2, padding=1, bias=True),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Dropout(.2),

        nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),

        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=True),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Dropout(.2),

        nn.Conv2d(32, 16, kernel_size=1, stride=2, padding=1, bias=True),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(9),
        nn.Flatten(),
        nn.Linear(16, out_dim, bias=True))
    return model

# -------------------------------------------------------------------------------------------------

################### Step 1 ###################
# We initialize the Experience Replay memory


class ReplayBuffer(object):

    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states1, batch_states2, batch_next_states1, batch_next_states2, batch_actions, batch_rewards, batch_dones = [], [], [], [], [], [], []
        for i in ind:
            state1, state2, next_state1, next_state2, action, reward, done = self.storage[i]
            # batch_states1.append(state1)
            # batch_states2.append(np.array(state2, copy=False))
            batch_states1.append(trans_form(state1).permute(0, 1, 2))
            batch_states2.append(np.array(state2, copy=False))
            # batch_next_states1.append(next_state1)
            # batch_next_states2.append(np.array(next_state2, copy=False))
            batch_next_states1.append(trans_form(next_state1).permute(0, 1, 2))
            batch_next_states2.append(np.array(next_state2, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return torch.stack([i for i in batch_states1]), np.array(batch_states2), torch.stack([i for i in batch_states1]), np.array(batch_next_states2), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

    def length(self):
        return len(self.storage)

# -------------------------------------------------------------------------------------------------

################## Step 2 ##################
# We build one neural network for the Actor model and one neural network for the Actor target
# Actor input is patch image (say 60x 60)
# Output is action (which is the orientation value for us)


class Actor(nn.Module):

    def __init__(self, action_dim, max_action):
        super(Actor, self).__init__()

        hidden_1_dim = 400
        hidden_2_dim = 300

        self.conv_img = ImageConv(1, 32)
        concat_input = 32 + 3

        self.layer_1 = nn.Linear(concat_input, hidden_1_dim, bias=True)
        self.layer_2 = nn.Linear(hidden_1_dim, hidden_2_dim, bias=True)
        self.layer_3 = nn.Linear(hidden_2_dim, action_dim, bias=True)

        # TODO: read about layernorm
        # NOT USED
        # self.layer_1_bn = nn.LayerNorm(hidden_1_dim)
        # self.layer_2_bn = nn.LayerNorm(hidden_2_dim)

        # for param in self.conv_img.parameters():
        #     param.requires_grad = False

    def forward(self, input_1, input_2):
        x1 = self.conv_img(input_1)
        input_2 = input_2
        x = torch.cat([x1, input_2], 1)
        x = F.elu(self.layer_1(x))
        x = F.elu(self.layer_2(x))
        # need to check the temperature value
        return max_action * torch.tanh(self.layer_3(x) * temperature)

# -------------------------------------------------------------------------------------------------
################## Step 3 ##################
# TODO comments about the critic network


class Critic(nn.Module):
    def __init__(self, action_dim):
        super(Critic, self).__init__()

        # convert image to a small dim
        # first dim --> number of input channel
        # 2nd dim --> output neurons

        hidden_1_dim = 400
        hidden_2_dim = 300

        self.conv_img = ImageConv(1, 32)
        state_dim = 32 + 3  # for orientation
        self.layer_1 = nn.Linear(
            state_dim + action_dim, hidden_1_dim, bias=True)
        self.layer_2 = nn.Linear(hidden_1_dim, hidden_2_dim, bias=True)
        self.layer_3 = nn.Linear(hidden_2_dim, 1, bias=True)

        # TODO: read about layernorm
        # NOT USED
        self.layer_1_bn = nn.LayerNorm(hidden_1_dim)
        self.layer_2_bn = nn.LayerNorm(hidden_2_dim)

        # Defining the second Critic neural network
        self.layer_4 = nn.Linear(
            state_dim + action_dim, hidden_1_dim, bias=True)
        self.layer_5 = nn.Linear(hidden_1_dim, hidden_2_dim, bias=True)
        self.layer_6 = nn.Linear(hidden_2_dim, 1, bias=True)
        # TODO: read about layernorm
        # NOT USED

        self.layer_4_bn = nn.LayerNorm(hidden_1_dim)
        self.layer_5_bn = nn.LayerNorm(hidden_2_dim)

        # for param in self.conv_img.parameters():
        #     param.requires_grad = False

    def forward(self, x_1, x_2, u):
        x1 = self.conv_img(x_1)
        x_2 = x_2
        xu = torch.cat([x1, x_2, u], 1)

        # Forward-Propagation on the first Critic Neural Network
        x1 = F.elu(self.layer_1(xu))
        x1 = F.elu(self.layer_2(x1))
        x1 = self.layer_3(x1)

        # Forward-Propagation on the second Critic Neural Network
        x2 = F.elu(self.layer_4(xu))
        x2 = F.elu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, x_1, x_2, u):
        x1 = self.conv_img(x_1)
        x_2 = x_2
        xu = torch.cat([x1, x_2, u], 1)

        # Forward-Propagation on the first Critic Neural Network
        x1 = F.elu(self.layer_1(xu))
        x1 = F.elu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

# -------------------------------------------------------------------------------------------------

################### Step 4 ###################
# Building the whole Training Process into a class


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv_layer_1 = ImageConv(1, 32)
#         self.final = nn.Linear(32, 1)

#     def forward(self, x):
#         x1 = self.conv_layer_1(x)
#         x1 = self.final(x1)
#         output = F.sigmoid(x1)
#         return output


# def copy_weights(cnn_image_classifier, model, device):
#     print('copying weights')
#     cnn_image_classifier = cnn_image_classifier.to(device)
#     for index, (param_cnn, param_model) in enumerate(zip(cnn_image_classifier.parameters(), model.parameters())):
#         if index < 13:
#             param_model.data.copy_(param_cnn)


class TD3(object):

    def __init__(self, action_dim, max_action):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # self.cnn_image_classifier = Net().to(self.device)
        # self.cnn_image_classifier.load_state_dict(
        #     torch.load('./cnn_weights.pt'))

        self.actor = Actor(action_dim, max_action).to(self.device)
        # copy_weights(self.cnn_image_classifier, self.actor, self.device)
        self.actor_target = Actor(action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr)
        self.critic = Critic(action_dim).to(self.device)
        # copy_weights(self.cnn_image_classifier,self.critic,self.device)
        self.critic_target = Critic(action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr)

    def select_action(self, state):
        input1 = trans_form(state[0]).permute(
            0, 1, 2).float().unsqueeze(0).to(self.device)
        input2 = torch.Tensor(state[1:4]).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            x = self.actor(input1, input2).cpu().data.numpy().flatten()
        return x

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        if iterations > 2000:
            iterations = 2000
        else:
            iterations = iterations

        for it in range(iterations):

            # We sample a batch of transitions (s, s’, a, r) from the memory
            batch_states1, batch_states2, batch_next_states1, batch_next_states2, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(
                batch_size)
            # print(torch.Tensor(batch_states1).unsqueeze(1).shape,batch_next_states1.shape)
            # print(torch.Tensor(batch_states1).shape,batch_next_states2.shape)
            # Test code for reply buffer:
            # for index in range(0, batch_size):

            #     state_last_img = np.squeeze(
            #         batch_states1[index].numpy() * 255)
            #     Image.fromarray(state_last_img).convert('L').save(
            #         os.path.join('./replay_buffer', 'current_{}.png'.format(index)))

            #     state_new_image = np.squeeze(
            #         batch_next_states1[index].numpy() * 255)
            #     Image.fromarray(state_new_image).convert('L').save(
            #         os.path.join('./replay_buffer', 'next_{}.png'.format(index)))

            # TODO remove prints
            # print(batch_states.shape, batch_next_states.shape, len(batch_actions), len(batch_rewards), len(batch_dones))
            # print(batch_next_states.shape)
            # print(batch_states2,batch_states2.shape)
            state_1 = batch_states1.float().to(self.device)
            state_2 = torch.Tensor(batch_states2).float().to(self.device)
            next_state_1 = batch_next_states1.float().to(self.device)
            next_state_2 = torch.Tensor(
                batch_next_states2).float().to(self.device)
            action = torch.Tensor(batch_actions).to(self.device)
            reward = torch.Tensor(batch_rewards).to(self.device)
            done = torch.Tensor(batch_dones).to(self.device)
            # print('----------------------------------------------------------')
            # print('state_1.shape, next_state_1.shape , state_1.min(),next_state_1.min(),state_1.max(),next_state_1.max() ---',
            #       state_1.shape, next_state_1.shape, state_1.min(), next_state_1.min(), state_1.max(), next_state_1.max())
            # print('------------')
            # print('state_2.shape,next_state_2.shape,action.shape,reward.shape,done.shape ---',
            #       state_2.shape, next_state_2.shape, action.shape, reward.shape, done.shape)
            # print('------------')
            # print('state_2.min,state_2.max,next_state_2.min,next_state_2.max,reward.min,reward.max ---',
            #       state_2.min(), state_2.max(), next_state_2.min(), next_state_2.max(), reward.min(),reward.max())
            # print('----------------------------------------------------------')

            # From the next state s’, the Actor target plays the next action a’
            next_action = self.actor_target(next_state_1, next_state_2)

            # We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
            noise = torch.Tensor(batch_actions).data.normal_(
                0, policy_noise).to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (
                next_action + noise).clamp(-max_action, max_action)

            # The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
            target_Q1, target_Q2 = self.critic_target(
                next_state_1, next_state_2, next_action)

            # We keep the minimum of these two Q-values: min(Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)

            # We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
            current_Q1, current_Q2 = self.critic(state_1, state_2, action)

            # We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(
                current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            if it % policy_freq == 0:
                actor_loss = - \
                    self.critic.Q1(state_1, state_2, self.actor(
                        state_1, state_2)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data)

                # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data)
            # TODO remove prints
            # print("check loss value", critic_loss, actor_loss)

    # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), os.path.join(
            directory, filename + '_actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(
            directory, filename + '_critic.pth'))

    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(
            '%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load(
            '%s/%s_critic.pth' % (directory, filename)))


# TODO move this block of constants up

########## CONSTANTS ##########
seed = 10  # Random seed number
start_timesteps = 1e4  # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
# How often the evaluation step is performed (after how many timesteps)
eval_freq = 5e3
max_timesteps = 5e5  # Total number of iterations/timesteps
save_models = True  # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1  # Exploration noise - STD value of exploration Gaussian noise
batch_size = 128  # Size of the batch
# Discount factor gamma, used in the calculation of the total discounted reward
discount = 0.99
tau = 0.005  # Target network update rate
# STD of Gaussian noise added to the actions for the exploration purposes
policy_noise = 0.2
# Maximum value of the Gaussian noise added to the actions (policy)
noise_clip = 0.5
policy_freq = 2
# Network Specific Dimensions
action_dim = 1
max_action = 5  # todo optimize with better value
temperature = 1
max_episode_steps = 5000

torch.manual_seed(seed)
np.random.seed(seed)

if not os.path.exists("./results"):
    os.makedirs("./results")


actor_lr = 0.0003
critic_lr = 0.0003

trans_form = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((.5,), (.5,))])

# -------------------------------------------------------------------------------------------------

################### Step 5 ###################
# Main training


class Train_TD3():

    def __init__(self, exp_name):

        # We set the parameters for first call
        # input_patch_size = [np.random.rand(1, 100, 100), 0, 0]
        # # self.last_state = torch.tensor(input_patch_size)
        # self.last_state = input_patch_size
        self.last_state = [Image.new('L', (60, 60)), 0., 0., 0.]
        self.last_action = np.array([0.])
        self.last_reward = 0.

        # Initialize Policy network
        self.policy = TD3(action_dim, max_action)
        self.replay_buffer = ReplayBuffer()

        # Initialize variables
        self.total_timesteps = 0
        self.episode_num = 0
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.done = 0.
        self.time_start = time.time()

        self.file_name = "%s_%s_%s" % ("TD3", exp_name, str(seed))
        print("---------------------------------------")
        print("Settings: %s" % (self.file_name))
        print("---------------------------------------")

        if save_models and not os.path.exists("./pytorch_models/" + self.file_name):
            os.makedirs("./pytorch_models/" + self.file_name)

    def update(self, reward, new_signal):
        # Input from ENV
        new_state, episode_done = new_signal[0], new_signal[1]

        # If done train the policy network
        if self.done:
            "Episode is over"
            # print("action predicted", action)
            # If we are not at the very beginning, we start the training process of the model
            if self.total_timesteps > 1000:
                print("Training started Total Timesteps: {} Episode Timesteps: {} Episode Num: {} Reward: {}".format(
                    self.total_timesteps, self.episode_timesteps, self.episode_num, self.episode_reward))
                self.policy.train(self.replay_buffer, self.episode_timesteps, batch_size,
                                  discount, tau, policy_noise, noise_clip, policy_freq)
                print("training done taking action")

            self.episode_reward = 0
            self.episode_timesteps = 0
            self.episode_num += 1
            self.done = False

        if self.total_timesteps % 1e4 == 0:
            print('saving models')
            self.policy.save('{}_{}'.format(
                self.file_name, self.episode_num), directory="./pytorch_models/" + self.file_name)

        self.replay_buffer.add(
            (self.last_state[0], self.last_state[1:4], new_state[0], new_state[1:4], self.last_action, self.last_reward, float(self.done)))
        # Action random or based on model prediction
        # if self.total_timesteps < start_timesteps or 30000 < self.total_timesteps < 32000 or 100000 < self.total_timesteps < 110000:
        #     action = np.random.uniform(-max_action, max_action, 1)
        if self.total_timesteps < start_timesteps:
            action = np.random.uniform(-max_action, max_action, 1)
        else:
            action = self.policy.select_action(new_state)
            # print('check action',action)
            if expl_noise != 0:
                action = (action + np.random.normal(0,
                                                    expl_noise, size=1)).clip(-1, 1)
        # print(action)
        # Add transition to replay buffer
        # print("before adding ", (new_state[0].size),type(new_state[1:4]),self.last_state[0].size,type(self.last_state[1:4]))
        # print('replay buffer ',self.last_action,self.last_reward, float(episode_done), self.last_state[1:4],new_state[1:4])

        self.done = float(episode_done)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward

        # update the tracking variables
        self.episode_reward += reward
        self.episode_timesteps += 1
        self.total_timesteps += 1

        return self.last_action
