# AI Method 2 - A* - Jinbiao Wang

# Table of Contents
- [AI Method 2 - A\* - Jinbiao Wang](#ai-method-2---a---jinbiao-wang)
- [Table of Contents](#table-of-contents)
    - [Motivation](#motivation)
    - [Application](#application)
    - [Trade-offs](#trade-offs)
      - [*Advantages*](#advantages)
      - [*Disadvantages*](#disadvantages)
    - [Challenges](#challenges)
    - [Future improvements](#future-improvements)



### Motivation  

In this Splendor game, the main motivation for using the A* algorithm to create agent is that it is very suitable for solving shortest path problems. In the strategy board game Splendor, players need to choose appropriate action to execute based on the current state of the game. It involves complex decision-making considerations. Different resource acquisition and usage sequences have a great impact on the victory of the game. The A* algorithm is very suitable for solving this type of problem because it can search through all states to return an optimal sequence of actions.

[Back to top](#table-of-contents)

### Application  

I first use the state-space model to model the Splendor game. State includes information about the current game, such as noble card information, the 12 cards on the table, and the gems available for collection. In addition, it also includes the current score of each player, the gems, and cards in each player's hand. Action contains collect_diff, collect_same, reserve, buy_available and buy_reserve. Initial State is the game_state in SelectAction(self, actions, game_state). Goal State is that my score is greater than or equal to 15 or the score of the current state is greater than the score of the initial state. The cost of each player action is 1.

My heuristic function  
![alt text](/wiki-template/images/H.png)

my_score and opponent_score are the current scores between myself and my opponent. my_noble_score and opponent_noble_score are based on how many cards in my and my opponent's hands meet noble’s cost requirement. my_card_points and opponent_card_points are the sum of the scores and quantities of my and my opponent's current cards. my_gems_score and opponent_gems_score are the current number of gems for my and my opponent. action_score is calculated using the action that reached the current state and the current state.

The Agent will use the actions and game_state in SelectAction(self, actions, game_state) to simulate the game and generate new nodes. The nodes contain game states and heuristic values. The Agent use the A* algorithm to search these nodes to find the shortest action execution sequence to get score. Then return the first action in the solution as our return action.

[Back to top](#table-of-contents)

### Trade-offs  
#### *Advantages*  
1. If the heuristic function is Admissible, the A* algorithm can find the shortest action execution sequence to obtain score.
2. The A* algorithm is very suitable for searching in complex environments. Therefore, it is very suitable for Splendor, a game that requires evaluating different decisions.

#### *Disadvantages*
1. Because Splendor game is too complex, the computational cost of the A* algorithm is very high.
2. The performance of the A* algorithm depends on the heuristic function. A bad heuristic function may reduce the performance of the Agent.

[Back to top](#table-of-contents)

### Challenges
Because the Splendor game is too complex, the A* algorithm may not be able to find the solution within 1 second. In order to ensure that the A* algorithm can calculate a result within 1 second, I wrote a filterAction function to filter useless actions to reduce the number of nodes required for calculation. For example, when there are no gems in hand, only collect action is considered. The reserve action is that I only consider reserve when I only have one gem to collect before I can buy this card. Even if actions are filtered out, the A* algorithm may not be able to calculate a result in one second. When the A* algorithm cannot return a result within one second, my Agent will select an action with the highest heuristic value from the open_list to return. In order to increase the number of times the A* algorithm can calculate the result, I added an action_score to the heuristic function. For example, the action used to reach the current state is collect gems. Add the collected gems and development card gems together to calculate the available gems. Then, calculate action_score by comparing the matching degree between the available gems and the 12 cards on the table. Adding this action_score allows the agent to choose gems that match the cost of the cards on the table every time, allowing the agent to buy cards faster and speed up the efficiency of A* calculation results.

[Back to top](#table-of-contents)

### Future improvements  
1. Use pruning technology to reduce the nodes that need to be searched, save computing time, hence improve the performance of the Agent.
2. Try more heuristic functions to improve Agent performance.

[Back to top](#table-of-contents)

