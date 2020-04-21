# Multitask_Alife

4/20/20 updates.

Next steps: 
(1) investigate further patterns in lesions, variance, and MI analysis.
MI seems to be a slightly better predictive measure, but its hard to say. The main differences are that Var and MI are reporting lower amounts of all three task involvement, and higher numbers of single and dual task involvement.

(2) Repeat all analysis for 4 tasks, adding the continuous mountaincar task. 

(3) Change network architecture, see if reducing the number of neurons/connections will result in higher levels of reuse, and vice versa. 

(4) add variations to the current tasks to make them more complex (ie, add more legs to the legged walker). Talked to Madhuven about how to train the network to work for two legs, without losing all of the knowledge gained from the previous evolvement for one leg. He suggested:

Step 1: Train X generations for 1 legged walker (inputs corresponding to the second leg would be set to 0)
Step 2: Continue EA and train Y generations for 1 legged walker and 2 legged walker.

The same could preseumably be done for adding more links on the inverted pendulum. Not sure about how to modify the cartpole task yet. 

(5) Find more predictive measures (apart from variance and mutual information) to analyze the neural traces. 
