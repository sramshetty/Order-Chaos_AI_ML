# Order-Chaos
Using Reinforcement Learning and Minimax to try and solve the game of Order and Chaos

### Resources used
* https://github.com/Cledersonbc/tic-tac-toe-minimax
* https://github.com/tansey/rl-tictactoe

By editing the above code I wanted to experiment with the limitations of minimax and q-learning on the game of Order and Chaos. Order and Chaos is a Tic-Tac-Toe derivative where the grid is 6-by-6 and a player wins with a 5 in a row. 
> Originally minimax was far too slow to even run due to the large depth of the grid, so I capped the depth at 3 when depth was greater than 10. Though this helped slighlty, I sought to further improve the runtime by implementing alpha-beta pruning. Doing so allowed me to reasonably increase my cap depth from 3 to 5, but not much further. Therefore, since all possibilities are usually not found, the AI makes some poor decisions or plays in defense for most of the game. 
>Then I tried q-learning on the game, expecting the table to get very large. This is exactly what happened, but training didn't need to be capped. So, on my 16gb ram machine, the table took up upwards of 96% of memory. Again, training did complete but the model did not perform well. I believe the limit on the number of episodes/games hindered the model from learning an optimal strategy in this game. The same number worked for Tic-Tac-Toe, but Tic-Tac-Toe is a much simpler game. 
