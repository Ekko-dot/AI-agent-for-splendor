from template import Agent 
import random
import copy
from Splendor.splendor_model import SplendorGameRule
import time

COLOURS = ['black','red','yellow','green','blue','white']

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.game_rule = SplendorGameRule(2)
        
    
    def SelectAction(self,actions,game_state):
        start_time = time.time()
        noble = game_state.board.nobles
        alpha = float('-inf')
        beta = float('inf')
        opponent_id = 1 - self.id
        filter_actions = self.filter_action(actions, game_state,self.id)
        max_value = -100000
        max_action = actions[0]
        depth = 1
        for action in actions:
            current_state = copy.deepcopy(game_state)
            new_state = self.game_rule.generateSuccessor(current_state,action,self.id)
            value = self.alphaBeta(depth-1, opponent_id,new_state, alpha, beta,noble, action)
            if(value >= max_value):
                max_value = value
                max_action = action
            if time.time() - start_time >= 0.8:
                #print(time.time() - start_time)
                return max_action
        # print(max_action)
        return max_action

    def alphaBeta(self,depth, player, gameState, alpha,beta,noble, my_action):
        totoal_score = self.game_rule.calScore(gameState, self.id)
        if depth == 0:
            # return self.evaluate_state(gameState,noble)
            return self.evaluate_state(gameState, my_action)
        elif player == self.id:
            actions = self.game_rule.getLegalActions(gameState, player)
            filter_actions = self.filter_action(actions,gameState,player)
            for action in filter_actions:
                current_state = copy.deepcopy(gameState)
                new_state = self.game_rule.generateSuccessor(current_state,action,player)
                value = self.alphaBeta(depth-1, 1 - player,new_state, alpha, beta,noble, action)
                alpha = max(value, alpha)
                if alpha >= beta:
                    break
            return alpha
        else:
            actions = self.game_rule.getLegalActions(gameState, player)
            filter_actions = self.filter_action(actions,gameState,player)
            # for action in filter_actions
            for action in actions:
                current_state = copy.deepcopy(gameState)
                new_state = self.game_rule.generateSuccessor(current_state,action,player)
                value = self.alphaBeta(depth-1, self.id,new_state, alpha, beta,noble, my_action)
                beta = min(value, beta)
                if alpha <= beta:
                    break
            return beta




    def filter_action(self, actions,gameState,player):
        num_gems = sum(gameState.agents[player].gems.values())
        buy_list = []
        collect_list = []
        resersed_list = []
        for action in actions:
            if action["type"] == 'buy_avaliable' or action["type"] == 'buy_reserve':
                buy_list.append(action)
            elif action["type"] == 'collect_diff' or action["type"] == 'collect_same':
                collect_list.append(action)
            else:
                resersed_list.append(action)
        if len(buy_list) >= 1:
            return buy_list
        if len(collect_list) >= 1 and num_gems <= 8 and sum(gameState.board.gems.values()) >= 3:
            return collect_list + resersed_list
        return resersed_list
        
    
    # def evaluate_state(self,gameState,noble):
    #     value6 = sum(gameState.agents[self.id].gems.values())
    #     value5 = len(gameState.agents[self.id].nobles)
    #     value1 = 0
    #     value3 = 0
    #     card_dict = gameState.agents[self.id].cards
    #     dict = gameState.agents[1 - self.id].cards
    #     value4 = 0
    #     for color in COLOURS:
    #         if color != "yellow":
    #             value1 += len(card_dict[color])
    #     for nb in noble:
    #         record = 0
    #         record2 = 0
    #         for key, value in nb[1].items():
    #             if key != "yellow":
    #                 value3 -= value
    #                 if len(card_dict[key]) > 0 and len(card_dict[key]) <= value:
    #                     value3 += (len(card_dict[key]))
    #                 if len(card_dict[key]) > 1:
    #                     record += 1
    #         if record > 1 and len(nb[1].keys()) == 2:
    #             value4 += 2
    #         if record > 1 and len(nb[1].keys()) == 3:
    #             value4 += 1
    #         if record > 2 and len(nb[1].keys()) == 3:
    #             value4 += 2
    #         #     if(len(card_dict[key]) >= value):
    #         #         record += 1
    #         # value3 += record
    #     #value2 = self.game_rule.calScore(gameState, self.id) - self.game_rule.calScore(gameState, 1 - self.id)
    #     #value2 = gameState.agents[self.id].score - gameState.agents[1 - self.id].score
    #     value2 = 0
    #     for color, cards in gameState.agents[self.id].cards.items():
    #         if color != "yellow":
    #             for card in cards:
    #                 value2 += card.points
    #     #value2 = self.game_rule.calScore(gameState, self.id)
    #     #print(value2*3 + value3*0.01 + value4)
    #     return value2*10 + value3*0.01 + value4*0.1 + value6*0.00001 + value1*0.0001
    


    def evaluate_state(self, state, action):
        # Current score
        my_score = 0
        opponent_score = 0

        opponent_id = 1 - self.id

        my_score += (state.agents[self.id].score)
        opponent_score += (state.agents[opponent_id].score)

        # Card score
        my_card_points = 0
        opponent_card_points = 0

        # for color, cards in state.agents[self.id].cards.items():
        #     if color != "yellow":
        #         for card in cards:
        #             my_card_points += card.points + 1
        
        # for color, cards in state.agents[opponent_id].cards.items():
        #     if color != "yellow":
        #         for card in cards:
        #             opponent_card_points += card.points + 1

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

        # for noble in state.board.nobles:
        #     for color, count in noble[1].items():
        #         if color != "yellow" and len(state.agents[self.id].cards[color]) != 0 and len(state.agents[self.id].cards[color]) <= noble[1][color]:
        #             my_noble_score += len(state.agents[self.id].cards[color])

        #         if color != "yellow" and len(state.agents[opponent_id].cards[color]) != 0 and len(state.agents[opponent_id].cards[color]) <= noble[1][color]:
        #             opponent_noble_score += len(state.agents[opponent_id].cards[color])

        # Gems score
        my_gems_score = sum(state.agents[self.id].gems.values())
        opponent_gems_score = sum(state.agents[opponent_id].gems.values())

        # Action score
        action_score = 0
        if "collect" in action["type"]:
            action_score = collectCheck(state, action, self.id)

        elif "buy" in action["type"]:
            action_score = buyCheck(state, action)

        # opponent_action_score = 0
        # if "collect" in action["type"]:
        #     opponent_action_score = collectCheck(state, opponent_action, opponent_id)

        # elif "buy" in action["type"]:
        #     opponent_action_score = buyCheck(state, opponent_action)

        # h = my_score + my_noble_score*0.1 + my_card_points*0.01 + my_gems_score*0.001
        # h = (my_score - opponent_score) + (my_noble_score - opponent_noble_score)*0.1 + (my_card_points - opponent_card_points)*0.01 + (my_gems_score - opponent_gems_score)*0.001
        h = (my_score - opponent_score) + (my_noble_score - opponent_noble_score)*0.1 + (my_card_points - opponent_card_points)*0.01 + action_score*0.001 + (my_gems_score - opponent_gems_score)*0.0001
        # h = (my_score - opponent_score) + action_score*0.1

        return h # 40%
        # return h # 40%

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