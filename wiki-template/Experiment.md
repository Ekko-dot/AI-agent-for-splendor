# Experiment

## Experiment method
We use 100 rounds of games as a benchmark, using each agent as the first player and second player to compete, and finally judge the performance of the Agent by comparing the game results.

## Result table
|                       |  Average Score  |  Winning Rate  |
| :-------------------: | :-------------: | :------------: |
| A* VS Original MinMax |  12.57 VS 13.96 |   39% VS 61%   |
| Original MinMax VS A* |  13.96 VS 12.39 |   61% VS 39%   |
| DQN VS Original MinMax|  4.59 VS 16.17 |   0% VS 100%   |
| Original MinMax VS DQN|  16.06 VS 4.46 |   100% VS 0%   |
| A* VS DQN             |  16.05 VS 5.07 |   100% VS 0%   |
| DQN VS A*             |  5.99 VS 16.09 |   1% VS 99%   |
| A* VS Final MinMax    |  12.51 VS 13.21 |   44% VS 55%   |
| Final MinMax VS A*    |  14.07 VS 11.97 |   60% VS 40%   |
| DQN VS Final MinMax   |  5.31 VS 16.27 |   0% VS 100%   |
| Final MinMax VS DQN   |  15.97 VS 5.12 |   99% VS 1%   |

## 
### A* VS Original MinMax
![alt text](/wiki-template/images/aStar-VS-MinMaxO.png)

### Original MinMax VS A*
![alt text](/wiki-template/images/MinMaxO-VS-aStar.png)

### DQN VS Original MinMax
![alt text](/wiki-template/images/DQN-VS-Original-MinMax.png)

### Original MinMax VS DQN
![alt text](/wiki-template/images/Original-MinMax-VS-DQN.png)

### A* VS DQN
![alt text](/wiki-template/images/aStar-VS-DQN.png)

### DQN VS A*
![alt text](/wiki-template/images/DQN-VS-aStar.png)

### A* VS Final MinMax 
![alt text](/wiki-template/images/aStar-VS-MinMax.png)

### Final MinMax VS A*
![alt text](/wiki-template/images/MinMax-VS-aStar.png)

### DQN VS Final MinMax
![alt text](/wiki-template/images/DQN-VS-Final-MinMax.png)

### Final MinMax VS DQN
![alt text](/wiki-template/images/Final-MinMax-VS-DQN.png)

## 
## Summary
At the beginning, by comparing three agents using different technologies to play against each other, based on the results, we found that the MinMax algorithm performed relatively well in this game. Therefore, we focus on improving the performance of the MinMax agent by modifying the heuristic function in the A* agent as the evaluation function in the MinMax agent. We found that this version of the MinMax agent performed better than the original three agents. So we choose this version of MinMax agent to compete on server.