 ## Variable importance and random forests 

## Which feature (also known as 'Variable') should you use to predict your target feature?

Examination of which of your data features are important in predicting your target variable achieved by a tool widely regarded as the best approach, Random Forest (RF) 
fit variable importance score. The algorithm's results are based on beta coefficients in logistic regression models and this tool is used for
dimensionality reduction. 

To give a simply analogy, if we have neurons and we recorded several parameters of their activity, we can use each of these 
candidate variables and evaluat them in relation to a binary outcome of interest, for instance whether they will 
show memory or no memory responses after being trained on such tasks. 

RF fit represents the effect of a variable in both main effects and interactions. Such RF variable 
importance is quantified to measure total decrease in RF tree node impurities. The method tabulates the 
top competing variables at a split since not all candidate and important variables are used at each split.
 
A mistake in categorizing an can be calculated as mean squared error (MSE) - as a measure of loss 
function reduction for each variable and split (returned as sum). In other words, MSE is used to measure 
the frequency of incorrect labelling of a randomly chosen element based on the labels. When a single 
category holds all the cases in a given node, i.e., when it is a pure tree, this error is at minimum (zero).


# Variable importance procedues in general:


1. First calculate the the effect of using all variables
2. Next start dropping/shuffling features and keep calculating prediction errors
3. Finally rank variables (AKA features) in order of decreasing importance (i.e. accuracy in prediction). For instance, it is possible that a neuron's firing rate is a better predictor of its memory potential, than, let's say its overall shape.
4. Show your results in a relational(proportional) and simple graphic display which shows the heirearchy, for instance horizonal bar charts where the top bar is the most influencial feature. 

 
Go here for more info: https://www.displayr.com/how-is-variable-importance-calculated-for-a-random-forest/
<p> </p>
<hr>
Thank you for trying the tool. Please feel free to contact me for feedback and questions.
<hr>
<p> </p>

©  Last edit March 26, 2021. Mulugeta Semework Abebe
