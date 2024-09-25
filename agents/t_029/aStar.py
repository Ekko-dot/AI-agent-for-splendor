from template import Agent
import random
import time
from Splendor.splendor_model import SplendorGameRule
import copy

CALCULATINGTIME = 0.9
NUMOFAGENT = 2

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
    
    def SelectAction(self,actions,game_state):
        start_time = time.time()
        game = SplendorGameRule(NUMOFAGENT)
        opponent_id = getOpponentId(self)
        start_state = copy.deepcopy(game_state)

        # Filter actions to obtain the most useful actions and reduce calculation time
        filtered_actions = filterAction(self.id,actions,start_state)

        open_list = []

        start_node = (start_state, 0, [], None)
        open_list.append((start_node, heuristic(start_state,self,{"type":""},0,opponent_id)))
        
        pre_score = start_state.agents[self.id].score

        while len(open_list) != 0 and time.time() - start_time < CALCULATINGTIME:
            node = open_list[0]
            open_list.pop(0)
            state, cost, action, initial_action = node[0]
            h_list = []

            if state.agents[self.id].score >= 15 or state.agents[self.id].score != pre_score:
                return initial_action
            
            pre_score = state.agents[self.id].score
            
            filtered_actions = filterAction(self.id, game.getLegalActions(state,self.id), state)

            for c_action in filtered_actions:
                current_state = copy.deepcopy(state)
                my_state = game.generateSuccessor(current_state, c_action, self.id)
                opponent_actions = filterAction(opponent_id, game.getLegalActions(my_state,opponent_id), my_state)

                for o_action in opponent_actions:
                    if time.time() - start_time > CALCULATINGTIME - 0.1:
                        open_list.sort(key=lambda x: x[1],reverse=True)
                        return open_list[0][0][3]
                    opponent_state = copy.deepcopy(my_state)
                    new_state = game.generateSuccessor(opponent_state, o_action, opponent_id)
                    new_cost = cost + 1
                    new_action = action + [c_action]
                    if initial_action == None:
                        new_node = (new_state, new_cost, new_action, c_action)
                        h = heuristic(new_state,self,c_action,new_cost,opponent_id)
                        h_list.append(h)
                    else:
                        new_node = (new_state, new_cost, new_action, initial_action)
                        h = heuristic(new_state,self,initial_action,new_cost,opponent_id)
                        h_list.append(h)
                    open_list.append((new_node, h))
                    open_list.sort(key=lambda x: x[1],reverse=True)

        open_list.sort(key=lambda x: x[1],reverse=True)
        return open_list[0][0][3]


def heuristic(state,self,action,cost,opponent_id):
    # Current score
    my_score = 0
    opponent_score = 0

    my_score += (state.agents[self.id].score)
    opponent_score += (state.agents[opponent_id].score)

    # Card score
    my_card_points = 0
    opponent_card_points = 0

    # Noble score
    my_noble_score = 0
    opponent_noble_score = 0

    noble_colour = []
    for noble in state.board.nobles:
        noble_colour += list(noble[1].keys())
    
    for color, cards in state.agents[self.id].cards.items():
        if color != "yellow" and color in noble_colour:
            my_noble_score += len(cards)
        if color != "yellow":
            for card in cards:
                my_card_points += card.points + 1

    for color, cards in state.agents[opponent_id].cards.items():
        if color != "yellow" and color in noble_colour:
            opponent_noble_score += len(cards)
        if color != "yellow":
            for card in cards:
                opponent_card_points += card.points + 1

    # Gems score
    my_gems_score = sum(state.agents[self.id].gems.values())
    opponent_gems_score = sum(state.agents[opponent_id].gems.values())

    # Action score
    action_score = 0
    if "collect" in action["type"]:
        action_score = collectCheck(state, action, self.id)

    elif "buy" in action["type"]:
        action_score = buyCheck(state, action)

    h = (my_score - opponent_score) + (my_noble_score - opponent_noble_score)*0.1 + (my_card_points - opponent_card_points)*0.01 + action_score*0.001 + (my_gems_score - opponent_gems_score)*0.0001

    return h - cost


def buyCheck(state, action):
    max_score = 0

    noble_colour = []
    for noble in state.board.nobles:
        noble_colour += list(noble[1].keys())
    noble_colour = set(noble_colour)

    if action["card"].colour in noble_colour:
        max_score = 8
    elif action["card"].points > 0:
        max_score = 7
    else:
        max_score = 0

    return max_score

def collectCheck(state, action, id):
    collected_gems = action["collected_gems"]
    my_gems = state.agents[id].cards
    current_gems = state.agents[id].gems

    available_gems = {}
    for colour, count in collected_gems.items():
        available_gems[colour] = count
        available_gems[colour] += len(my_gems[colour])
        available_gems[colour] += current_gems[colour]

    noble_colour = []
    for noble in state.board.nobles:
        noble_colour += list(noble[1].keys())
    noble_colour = set(noble_colour)

    max_match = 0
    card_list = []
    level_one_card = []

    for card in state.board.dealt_list():
        if card.colour in noble_colour:
            card_list.append(card)
        if card.deck_id == 0:
            level_one_card.append(card)

    extra_gem_colour = []
    e_max = 0

    for card in card_list:
        card_cost = card.cost
        current_match = 0
        for colour, count in card_cost.items():
            if colour in available_gems and card_cost[colour] - available_gems[colour] <= 0:
                current_match += 1
                if card_cost[colour] - available_gems[colour] == 0:
                    extra_gem_colour.append(colour)
        if current_match >= len(card_cost):
            for e_card in level_one_card:
                e_current_match = 0
                if e_card != card:
                    e_card_cost = e_card.cost
                    for colour, count in e_card_cost.items():
                        if colour not in extra_gem_colour and colour in available_gems:
                            e_current_match += 1
                if e_current_match > e_max:
                    e_max = e_current_match
        if current_match > max_match:
            max_match = current_match

    return max_match + e_max

def getSuccessors(state,actions,my_id,opponent_id,game,start_time):
    all_successors = []

    for action in actions:
        current_state = copy.deepcopy(state)
        my_state = game.generateSuccessor(current_state, action, my_id)
        opponent_actions = filterAction(opponent_id, game.getLegalActions(my_state,opponent_id), my_state)

        for o_action in opponent_actions:
            if time.time() - start_time > CALCULATINGTIME - 0.1:
                return all_successors
            opponent_state = copy.deepcopy(my_state)
            new_state = game.generateSuccessor(opponent_state, o_action, opponent_id)
            all_successors.append((new_state,action))
    return all_successors




# Use to obtain useful actions from actions
def filterAction(self_id,actions,game_state):
    collect_actions = []
    reserve_actions = []
    buy_actions = []
    pass_actions = []

    for action in actions:
        if "collect" in action["type"]:
            collect_actions.append(action)
        elif "reserve" == action["type"]:
            reserve_actions.append(action)
        elif "buy" in action["type"]:
            buy_actions.append(action)
        else:
            pass_actions.append(action)

    # Get current agent state
    agent_state = game_state.agents[self_id]

    # Get the current number of gems
    gems_count = sum(agent_state.gems.values())

    # If the current number of gems is 0, only consider collect action
    if gems_count == 0:
        return buy_actions + collect_actions + pass_actions
    
    # Get current gems
    current_gems = agent_state.gems

    # Get card list
    card_list = game_state.board.dealt_list()

    # Find cards we can reserve
    filtered_reserve_actions = []
    for card in card_list:
        card_cost = card.cost
        reserve = True
        for gem in card_cost:
            cost = card_cost[gem] - current_gems[gem]
            if cost > 1:
                reserve = False
        if reserve == True:
            for action in reserve_actions:
                if action["card"] == card:
                    filtered_reserve_actions.append(action)
    
    return buy_actions + collect_actions + filtered_reserve_actions + pass_actions
        
def getOpponentId(self):
    if self.id == 0:
        return 1
    elif self.id == 1:
        return 0
    else:
        return None