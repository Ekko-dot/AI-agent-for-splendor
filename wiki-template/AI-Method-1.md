# AI Method 1 - MinMax - Rui Peng

# Table of Contents
- [AI Method 1 - MinMax - Rui Peng](#ai-method-1---minmax---rui-peng)
- [Table of Contents](#table-of-contents)
    - [Motivation](#motivation)
    - [Application](#application)
    - [Trade-offs](#trade-offs)
      - [*Advantages*](#advantages)
      - [*Disadvantages*](#disadvantages)
    - [Challenges](#challenges)
    - [Future improvements](#future-improvements)



### Motivation  
The Splender (two players) can be considered as a zero-sum game because the number of development cards and nobles is limited. If a player gets a noble or a development card with a score can create an advantage in the game and it can be considered as a loss for another player because he/she can not get this noble and development card anymore. We consider that the MinMax algorithm can be one of the efficient techniques to handle the zero-sum game.

[Back to top](#table-of-contents)

### Application 
Based on the MinMax algorithm, the agent will simulate a certain round (it plays one action and its opponent plays one action) of the game and select the best action by considering the opponent would make a rational action as well. For the round simulation, we implement the getLegalActions() function to get all the possible actions of the two players and the generateSuccessor() to get the successor state based on the action. The function evaluate_state() evaluates the current state based on the state information. The final version of the evaluate function is similar to the evaluate function that is mentioned in the A* algorithm. The function provides an accurate evaluation of the current state, considering the player's current score, the number of nobles visited, the number of effective development cards (development cards with the bonus that are in noble’s requirement), and the current number of effective gems (gems that is useful to buy effective development card).

[Back to top](#table-of-contents)

### Trade-offs  
#### *Advantages*  
The advantage of a MinMax agent can make an optimal action by predicting the future actions of itself and its opponents. Compared to other agents, it is more likely to make decisions that yield greater long-term benefits, rather than focusing on short-term gains like most agents. Additionally, it can anticipate the opponent's moves and make counter-moves accordingly.

#### *Disadvantages*
On the other hand, if the opponent is irrational, the performance of the MinMax algorithm can degrade because it might overestimate certain actions. For example, it might predict that the opponent will buy a specific red development card, so it reserves that card preemptively. However, if the opponent does not buy that card, then this action becomes unnecessary.

[Back to top](#table-of-contents)

### Challenges
The main challenge is to design an efficient evaluation function that can evaluate the state accurately. The evaluation function directly impacts the performance of the MinMax agent. The first evaluation function only considers the current score, its performance is terrible, and sometimes it can not even defeat the random agent. We found that the main issue lies in only considering the current score, which cannot accurately analyze the current state. For example, when the agent buys a development card with no points versus not buying it, the current score remains the same. However, purchasing this development card can be more advantageous as it can help the agent acquire nobles or higher-point development cards later on.
Therefore, our evaluation function should consider multiple factors to accurately assess the advantages and disadvantages of each state.

[Back to top](#table-of-contents)

### Future improvements  
For improvement, the evaluation function still needs to improve. Our evaluation function cannot accurately assess the advantage of a state in all situations because, in the A* algorithm, its input is an agent's action and the resulting state after the agent completes the action. However, in our minimax algorithm, its input is an agent's action and the resulting state after both the agent and the opponent have executed their actions. This introduces an error in the calculation. For example, the evaluation function might include a value that calculates the advantage of a state based on the agent targeting a specific development card. If the opponent buys that card in the meantime, the action will be overestimated. We should adjust the calculation of this value, taking into account that the opponent might also purchase the target card by adding a condition to determine whether the opponent can buy our target card.

[Back to top](#table-of-contents)
