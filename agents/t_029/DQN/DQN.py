from template import Agent
import copy
from Splendor.splendor_model import SplendorGameRule
import random
import random,itertools
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
from Splendor.splendor_utils import CARDS
import itertools
import numpy as np
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
import torch.nn.init as init



Device = "cuda"
NUMOFAGENT = 2

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.model_path = 'agents/t_029/DQN/reward_nostep.pth'
        self.model = None
    
    def SelectAction(self,actions,game_state):
        
        if self.model == None:
            self.model = DQNNetwork(229,991)
            self.model.load_state_dict(torch.load(self.model_path,map_location=torch.device('cpu')))
            
            all_actions = create_all_actions(CARDS)
            action = generate_action(all_actions,actions,game_state,self.model)
            # print(time.time() - start)
            return action
        else:
            all_actions = create_all_actions(CARDS)
            action = generate_action(all_actions,actions,game_state,self.model)
            return action
        # !!!!!if you want to train, fix the code!!!!!
        # print('no model, need to train')
        # main(self.id,self.model_path)
    
def main(id,model_path):
# Game and network hyperparameters
    save_path = model_path
    state_dim = 229
    action_dim = 991
    model = DQNNetwork(state_dim, action_dim).to(Device)
    replay_buffer = ReplayBuffer(10000,model)
    optimizer = optim.Adam(model.parameters(), lr= 1e-7)
    game = SplendorGameRule(NUMOFAGENT)
    target_model = DQNNetwork(state_dim, action_dim).to(Device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    agent = DQNAgent(state_dim, action_dim, replay_buffer, model, target_model, optimizer)
    all_actions = create_all_actions(CARDS)
    
    losses = []

    num_episodes = 50
    batch_size = 32
    done = False

    for episode in range(num_episodes):
        state = game.initialGameState()
        last_state = copy.deepcopy(state)
        total_loss = 0
        step = 0
        random_sele = True
        done = False
        while not done:
            actions = game.getLegalActions(state, id)
            valid_actions_mask = valid_action(all_actions,actions)
            en_state = encode_game_state(state)
            if sum(valid_actions_mask) != 0:
                index = agent.select_action(en_state, valid_actions_mask)
                action = all_actions[index]
                for raction in actions:
                    if raction.get('type') == action.get('type'):
                        if raction.get('type') == 'reserve':
                            if raction.get('card').code == action.get('card') and raction.get('collected_gems') == action.get('collected_gems') and raction.get('returned_gems') == action.get('returned_gems'):
                                next_state = game.generateSuccessor(state,raction,id)
                                otr_action = game.getLegalActions(next_state, 1)
                                # Randomized agent confrontation training: 
                                # next_state = game.generateSuccessor(next_state,random.choice(otr_action),1)
                                # Self-confrontation training
                                next_state = game.generateSuccessor(next_state,generate_action(all_actions,otr_action,next_state,model),1)
                                random_sele = False
                        elif raction.get('type') == 'buy_available' or  raction.get('type') == 'buy_reserve':
                            if raction.get('card').code == action.get('card'):
                                next_state = game.generateSuccessor(state,raction,id)
                                otr_action = game.getLegalActions(next_state, 1)
                                # next_state = game.generateSuccessor(next_state,random.choice(otr_action),1)
                                next_state = game.generateSuccessor(next_state,generate_action(all_actions,otr_action,next_state,model),1)
                                random_sele = False
                        elif raction.get('type') == 'collect_diff' or raction.get('type') == 'collect_same':
                            if raction.get('collected_gems') == action.get('collected_gems') and raction.get('returned_gems') == action.get('returned_gems'):
                                next_state = game.generateSuccessor(state,raction,id)
                                otr_action = game.getLegalActions(next_state, 1)
                                # next_state = game.generateSuccessor(next_state,random.choice(otr_action),1)
                                next_state = game.generateSuccessor(next_state,generate_action(all_actions,otr_action,next_state,model),1)
                                random_sele = False
                        elif raction.get('type') == 'pass':
                            next_state = game.generateSuccessor(state,raction,id)
                            otr_action = game.getLegalActions(next_state, 1)
                            # next_state = game.generateSuccessor(next_state,random.choice(otr_action),1)
                            next_state = game.generateSuccessor(next_state,generate_action(all_actions,otr_action,next_state,model),1)
                            random_sele = False
                    
            if random_sele:
                raction = random.choice(actions)
                next_state = game.generateSuccessor(state,raction,id)
                otr_action = game.getLegalActions(next_state, 1)
                next_state = game.generateSuccessor(next_state,random.choice(otr_action),1)

            step += 1
            en_next_state = encode_game_state(next_state)
            reward, done = calculate_reward(last_state, next_state, id, step,raction)
            replay_buffer.push(en_state, index, reward, en_next_state, done)

            loss = agent.update_policy(replay_buffer,batch_size)
            if loss is not None:
                total_loss += loss
            
            stepp = step

        agent.update_epsilon()
        print(f"Episode {episode}: Total loss {total_loss/stepp} step {stepp}")
        losses.append(total_loss/stepp)

    # Saving model parameters
    torch.save(model.state_dict(), save_path)
    print(f"Model parameters saved to {save_path}")
    # image(losses)
    
def calculate_reward(state, next_state, id, step,raction):
    next_agent_state = next_state.agents[id]
    agent_state = state.agents[id]
    new_gem = 0
    old_gem = 0
    score = 0

    if next_agent_state.score >= 15:
        return (score + 300), True
    if next_agent_state.score > agent_state.score:
        score += (next_agent_state.score - agent_state.score)*15

    if raction.get('type') == 'buy_available' or raction.get('type') == 'buy_reserve':
        card = raction.get('card')
        if card.deck_id == 1:
            score += 10
        elif card.deck_id == 2:
            score += 20
        elif card.deck_id == 3:
            score += 70

    if len(next_agent_state.nobles) > len(agent_state.nobles):
        score +=  (len(next_agent_state.nobles) - len(agent_state.nobles)) * 50

    for num in next_agent_state.gems.values():
        new_gem += num
    for num in agent_state.gems.values():
        old_gem += num

    if new_gem >= old_gem and next_agent_state.score <= agent_state.score:
        score -= (new_gem - old_gem)*2

    if step > 40:
        score -= 100

    if step >= 100:
        return score , True
        # return score - 300 , True
    return score , False

    # next_agent_state = next_state.agents[id]
    # agent_state = state.agents[id]
    # new_gem = 0
    # old_gem = 0
    # score = 0

    # if next_agent_state.score >= 15:
    #     return (score + 300), True
    # if next_agent_state.score > agent_state.score:
    #     if step <= 5:
    #         score += (next_agent_state.score - agent_state.score)* 5
    #     elif step > 5 and step <= 15:
    #         score += (next_agent_state.score - agent_state.score)* 10
    #     elif step > 15 and step <= 25:
    #         score += (next_agent_state.score - agent_state.score)* 20

    # if raction.get('type') == 'buy_available' or raction.get('type') == 'buy_reserve':
    #     card = raction.get('card')
    #     if card.deck_id == 1 :
    #         if step <= 12:
    #             score += 50
    #         elif step > 12:
    #             score += 5
    #         elif step > 25:
    #             score += 1
    #     elif card.deck_id == 2 and step < 10:
    #         if step <= 12:
    #             score += 10
    #         elif step > 12 and step <= 25:
    #             score += 20
    #         elif step > 25:
    #             score += 10
    #     elif card.deck_id == 3:
    #         if step <= 12:
    #             score += 5
    #         elif step > 12 and step <= 20:
    #             score += 20
    #         elif step > 25:
    #             score += 70

    # if len(next_agent_state.nobles) > len(agent_state.nobles):
    #     if step >= 15 and step < 25:
    #         score +=  (len(next_agent_state.nobles) - len(agent_state.nobles)) * 100
    #     if step > 25:
    #         score +=  (len(next_agent_state.nobles) - len(agent_state.nobles)) * 50

    # for num in next_agent_state.gems.values():
    #     new_gem += num
    # for num in agent_state.gems.values():
    #     old_gem += num
    # if step < 10:
    #     if new_gem >= old_gem and next_agent_state.score <= agent_state.score:
    #         score -= new_gem - old_gem
    # if step >= 10 and step < 20:
    #      if new_gem >= old_gem and next_agent_state.score <= agent_state.score:
    #         score -= (new_gem - old_gem)*2
    # if step >= 20 and step < 30:
    #      if new_gem >= old_gem and next_agent_state.score <= agent_state.score:
    #         score -= (new_gem - old_gem)*5
    # if step >= 30 and step < 40:
    #      if new_gem >= old_gem and next_agent_state.score <= agent_state.score:
    #         score -= (new_gem - old_gem)*10


    # if step > 40:
    #     score -= 100

    # if step >= 100:
    #     # print(score)
    #     return score - 300 , True
    #     # score -= 2
    # return score , False


class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 768)
        self.fc3 = nn.Linear(768, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.output = nn.Linear(1024, action_dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.4)
        # you can initialize if you want
        # self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        return self.output(x)
    
    def _initialize_weights(self):
        init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.fc1.bias, 0)
        init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.fc2.bias, 0)
        init.kaiming_normal_(self.fc3.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.fc3.bias, 0)
        init.kaiming_normal_(self.fc4.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.fc4.bias, 0)

class ReplayBuffer:
    def __init__(self, capacity, model, n_steps=20):
        self.model = model
        self.n_steps = n_steps
        self.nbuffer = deque(maxlen=n_steps)
        self.buffer = deque(maxlen=capacity)
        self.cumulative_reward = 0 #acumulative reward during multi step method

    def push(self, state, action, reward, next_state, done):
        self.cumulative_reward = self.cumulative_reward * 0.99 + reward
        self.nbuffer.append((state, action, reward, next_state, done))
        if len(self.nbuffer) == self.n_steps or done:
            state_n, action_n, _, _, _ = self.nbuffer[0]
            _, _, _, next_state_n, done_n = self.nbuffer[-1]
            self.buffer.append((state_n, action_n, self.cumulative_reward, next_state_n, done_n))
            self.reset()
        
    def reset(self):
        self.cumulative_reward = 0
        self.nbuffer.clear()

    def sample(self, batch_size):

        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def len(self):
        return len(self.buffer)
    
    ## traditional ReplayBuffer without multi step method
    # def push(self, state, action, reward, next_state, done):
    #     self.buffer.append((state, action, reward, next_state, done))

    # def sample(self, batch_size):
    #     state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
    #     return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

class DQNAgent:
    def __init__(self, state_dim, action_dim, replay_buffer, model, target_model, optimizer,epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,tau=0.001,burn_in=50000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.epsilon = epsilon  # initial epsilon
        self.epsilon_min = epsilon_min  # min epsilon 
        self.epsilon_decay = epsilon_decay  # decay rate of epsilon 
        self.tau = tau #ONLY USE IN SOFT UPDATE
        self.step_count = 0
        self.burn_in = burn_in

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min  and self.step_count > self.burn_in:
            print('cahge')
            self.epsilon *= self.epsilon_decay

    def select_action(self, state, vmask):
        self.step_count += 1
        if random.random() < self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(Device)
            with torch.no_grad():
                output = self.model(state)
            vmask = torch.from_numpy(vmask).unsqueeze(0)
            output[~vmask] = float('-inf')
            index = torch.argmax(output).item()
        else:
            available_actions = [index for index, available in enumerate(vmask) if available]
            index = random.choice(available_actions)
        
        return index

    def update_policy(self, replay_buffer, batch_size):
        if replay_buffer.len() < batch_size:
            return None
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        scaler = StandardScaler()
        reward_std = scaler.fit_transform(reward.reshape(-1, 1)).flatten()
        state = torch.FloatTensor(state).to(Device)
        next_state = torch.FloatTensor(next_state).to(Device)
        action = torch.LongTensor(action).to(Device)
        reward = torch.FloatTensor(reward_std).to(Device)
        done = torch.FloatTensor(done).to(Device)

        current_q_values = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_state).max(1)[0]
        expected_q_values = reward + 0.99 * next_q_values * (1 - done)
        
        loss = torch.nn.functional.smooth_l1_loss(current_q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.soft_update(self.model, self.target_model, self.tau)


        return loss.item()
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    ## Exact replication of local model parameters to the target model
    # def hard_update(self, local_model, target_model):
    #     target_model.load_state_dict(local_model.state_dict())

def create_all_actions(CARDS):
            '''
            create all the actions possible
            '''
            actions = []
            
            gems = ["red", "blue", "white", "black", "green"]
            # output all possible gem fetching actions
    
            # Fetch different colored stones
            for num in range(1, 4):
                for gem_combo in itertools.combinations(gems, num):
                    if (num == 3):
                        for return_num in range(0, 2):
                            for returned_gem_combo in itertools.combinations([item for item in gems if item not in gem_combo], return_num):
                                actions.append({"type": "collect_diff", "collected_gems": {gem: 1 for gem in gem_combo}, "returned_gems": {
                                        gem: 1 for gem in returned_gem_combo}, "noble": None})
                    else:
                        for return_num in range(0, 3):
                            for returned_gem_combo in itertools.combinations([item for item in gems if item not in gem_combo], return_num):
                                actions.append({"type": "collect_diff", "collected_gems": {gem: 1 for gem in gem_combo}, "returned_gems": {
                                        gem: 1 for gem in returned_gem_combo}, "noble": None})
            # Gems of the same color
            for i in gems:
                for return_num in range(0, 2):
                    for returned_gem_combo in itertools.combinations([item for item in gems if item != i], return_num):
                        actions.append({"type": "collect_same", "collected_gems": {i: 2}, "returned_gems": {
                                gem: 1 for gem in returned_gem_combo}, "noble": None})

            # Purchase Cards
            for card in CARDS.keys():
                actions.append({'type': 'buy_available', 'card': card, 'returned_gems': CARDS[card][1],
                        'noble': None})
                actions.append({'type': 'buy_reserve', 'card': card, 'returned_gems': CARDS[card][1],
                        'noble': None})
                actions.append({'type': 'reserve', 'card': card, 'collected_gems': {
                        "yellow": 1}, 'returned_gems': {}, 'noble': None})
                actions.append({'type': 'reserve', 'card': card, 'collected_gems': {},
                        'returned_gems': {}, 'noble': None})
                # reserve card
                for i in gems:
                    actions.append({'type': 'reserve', 'card': card, 'collected_gems': {
                            "yellow": 1}, 'returned_gems': {i: 1}, 'noble': None})
                    
            actions.append({'type':'pass'})

            return actions


def encode_game_state(game_state):
    board_state = game_state.board
  
    # gem type
    gem_types = ['black', 'red', 'yellow', 'green', 'blue', 'white']
    cos_gem_type = ['black', 'red', 'green', 'blue', 'white']
    # artribute of card (tier,color,point,cost)
    tire_attri = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
    card_attributes = {
        'Tier 1': [1, 0, 0],
        'Tier 2': [0, 1, 0],
        'Tier 3': [0, 0, 1],
        'black': [1, 0, 0, 0, 0, 0],
        'red': [0, 1, 0, 0, 0, 0],
        'yellow': [0, 0, 1, 0, 0, 0],
        'green': [0, 0, 0, 1, 0, 0],
        'blue': [0, 0, 0, 0, 1, 0],
        'white': [0, 0, 0, 0, 0, 1]
    }
    
    # init vector
    gem_vector = np.zeros(len(gem_types))
    card_vector = []
    agent_vectors = []
    noble_vectors = []

    for color, num in board_state.gems.items():
        if color in gem_types:
            gem_index = gem_types.index(color)
            gem_vector[gem_index] = int(num)

    for card in board_state.dealt_list():
        tier = card.deck_id
        points = card.points
        colour = card.colour
        cost = card.cost
        cost_vector = np.zeros(len(cos_gem_type))
        for color, num in cost.items():
            cost_index = cos_gem_type.index(color)
            cost_vector[cost_index] = int(num)

        card_vector.extend(tire_attri[tier] + card_attributes[colour] + [points] + list(cost_vector))
     # agent state
    for i in range(NUMOFAGENT):
        Agent_state = game_state.agents[i]

        score = Agent_state.score
        gems = [Agent_state.gems.get(gem, 0) for gem in gem_types]
        cards = [len(Agent_state.cards.get(gem, [])) for gem in gem_types] 
        nobles_count = len(Agent_state.nobles)
        agent_vector = [score] + gems + cards + [nobles_count]
        agent_vectors.extend(agent_vector)

    # noble state
    for noble in board_state.nobles:
        requirements = [noble[1].get(gem, 0) for gem in cos_gem_type]
        noble_vectors.extend(requirements) 

    card_vector = np.array(card_vector)
    agent_vectors = np.array(agent_vectors)
    noble_vectors = np.array(noble_vectors)

    gem_vector = pad_with_zeros(gem_vector,6)
    card_vector = pad_with_zeros(card_vector,180)
    agent_vectors = pad_with_zeros(agent_vectors,28)
    noble_vectors = pad_with_zeros(noble_vectors,15)

    # concate them all
    full_vector = np.concatenate((gem_vector , card_vector , agent_vectors , noble_vectors))
    scaler = StandardScaler()

    full_vector_standardized = scaler.fit_transform(full_vector.reshape(-1, 1)).flatten()

    return full_vector_standardized

# pic loss
# def image(losses):
#     plt.figure(figsize=(10, 6))
#     plt.plot(losses, label='Loss per Episode', marker='o')
#     plt.title('Loss vs. Episodes')
#     plt.xlabel('Episode')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig('loss_nostep_mini.png', dpi=300)  
#     plt.show()

#padding functioni used in encode_game_state
def pad_with_zeros(vector, target_length):
    padding_length = target_length - len(vector)
    if padding_length > 0:
        padding = np.zeros(padding_length, dtype=vector.dtype)
        return np.concatenate((vector, padding))
    else:
        return vector

# convert output of nn into leagal actioins
def generate_action(all_actions,actions,game_state,model):
    model.eval()
    vmask = valid_action(all_actions,actions)
    input = encode_game_state(game_state)
    # if you want to train, need to trans state to cuda
    # state = torch.FloatTensor(input).unsqueeze(0).to(Device)
    state = torch.FloatTensor(input).unsqueeze(0)
    with torch.no_grad():
        output = model(state)
    vmask = torch.from_numpy(vmask).unsqueeze(0)
    output[~vmask] = float('-inf')
    index = torch.argmax(output).item()

    action = all_actions[index]
    
    for raction in actions:
        if raction.get('type') == action.get('type'):
            if raction.get('type') == 'reserve':
                if raction.get('card').code == action.get('card') and raction.get('collected_gems') == action.get('collected_gems') and raction.get('returned_gems') == action.get('returned_gems'):
                    return raction
            elif raction.get('type') == 'buy_available' or  raction.get('type') == 'buy_reserve':
                if raction.get('card').code == action.get('card'):
                    return raction
            elif raction.get('type') == 'collect_diff' or raction.get('type') == 'collect_same':
                if raction.get('collected_gems') == action.get('collected_gems') and raction.get('returned_gems') == action.get('returned_gems'):
                    return raction
            elif raction.get('type') == 'pass':
                return raction
        
    print('random!!!')
    return random.choice(actions)

# generate a legally functional mask 
def valid_action(all_actions,actions):
        valid_action = []
        for all_action in all_actions:
            valid_action.append(False)
            for avi_action in actions:
                if avi_action.get('type') == all_action.get('type'):
                    if avi_action.get('type') == 'reserve':
                        if avi_action.get('card').code == all_action.get('card') and avi_action.get('collected_gems') == all_action.get('collected_gems') and avi_action.get('returned_gems') == all_action.get('returned_gems'):
                            valid_action.pop()
                            valid_action.append(True)
                            break
                    elif avi_action.get('type') == 'buy_available' or  avi_action.get('type') == 'buy_reserve':
                        if avi_action.get('card').code == all_action.get('card'):
                            valid_action.pop()
                            valid_action.append(True)
                            break
                    elif avi_action.get('type') == 'collect_diff' or avi_action.get('type') == 'collect_same':
                        if avi_action.get('collected_gems') == all_action.get('collected_gems') and avi_action.get('returned_gems') == all_action.get('returned_gems'):
                            valid_action.pop()
                            valid_action.append(True)
                            break
                    elif avi_action.get('type') == 'pass':
                        valid_action.pop()
                        valid_action.append(True) 
                        break

        valid_array = np.array(valid_action)
        return valid_array