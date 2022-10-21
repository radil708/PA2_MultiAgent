# PA2_MultiAgent

This repo is meant to showcase my work on applying the following AI algorithms: minimax, expectimax, and alpha-beta pruning
for a pacman game.

Many files have already been provided. 
My goal is to implement methods/functions in the multiagents.py file

The autograder can be run using the command: python autograder.py
The autograder can be run on any specific question using the command: python autograder.py -q q2
The autograder can be run on any specific question without graphics using: python autograder.py -q q1 --no-graphics

#########################################################################################
Questions:

Reflex Agent
Q1.) Improve the ReflexAgent in multiAgents.pyto play respectably. The provided reflex agent code provides 
some helpful examples of methods that query the GameState for information. A capable reflex agent will
have to consider both food locations and ghost locations to perform well. Your agent should easily and 
reliably clear the testClassic layout:

python pacman.py -p ReflexAgent -l testClassic

-------------------------------------------------------------------------------------------

Minimax
Q2.) Now you will write an adversarial search agent in the provided MinimaxAgent class stub in multiAgents.py.
Your minimax agent should work with any number of ghosts, so you’ll have to write an algorithm that is 
slightly more general than what you’ve previously seen in lecture. In particular, your minimax tree will 
have multiple min layers (one for each ghost) for every max layer.

- The minimax values of the initial state in the minimaxClassic layout are 9, 8, 7, -492 for depths 
1, 2, 3 and 4 respectively. Note that your minimax agent will often win (665/1000 games for us) 
despite the dire prediction of depth 4 minimax.

python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
-------------------------------------------------------------------------------------------
AlphaBeta Pruning
Q3.) Make a new agent that uses alpha-beta pruning to more efficiently explore the minimax tree, in AlphaBetaAgent. 
Again, your algorithm will be slightly more general than the pseudocode from lecture, so part of the challenge 
is to extend the alpha-beta pruning logic appropriately to multiple minimizer agents.

You should see a speed-up (perhaps depth 3 alpha-beta will run as fast as depth 2 minimax). 
Ideally, depth 3 on smallClassic should run in just a few seconds per move or faster.

python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
-------------------------------------------------------------------------------------------

Expectimax 
Q4.) Minimax and alpha-beta are great, but they both assume that you are playing against 
an adversary who makes optimal decisions. As anyone who has ever won tic-tac-toe can tell
you, this is not always the case. In this question you will implement the ExpectimaxAgent,
which is useful for modeling probabilistic behavior of agents who may make suboptimal choices.

python autograder.py -q q4
-------------------------------------------------------------------------------------------
Evaluation Function
Q5.)
Write a better evaluation function for pacman in the provided function betterEvaluationFunction. 
The evaluation function should evaluate states, rather than actions like your reflex agent 
evaluation function did. With depth 2 search, your evaluation function should clear the 
smallClassic layout with one random ghost more than half the time and still run at a 
reasonable rate (to get full credit, Pacman should be averaging around 1000 points when he’s winning).

Grading: the autograder will run your agent on the smallClassic layout 10 times. We will assign p
oints to your evaluation function in the following way:

If you win at least once without timing out the autograder, you receive 1 points. Any agent not satisfying these criteria will receive 0 points.
+1 for winning at least 5 times, +2 for winning all 10 times
+1 for an average score of at least 500, +2 for an average score of at least 1000 (including scores on lost games)
+1 if your games take on average less than 30 seconds on the autograder machine, when run with --no-graphics.
The additional points for average score and computation time will only be awarded if you win at least 5 times.

python autograder.py -q q5
