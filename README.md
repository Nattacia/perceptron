# perceptron
perceptron lab


#  Excercise 1 : Questions and Answers

# 1. What does the prediction -1 mean for the book [size=3, color=2]?
The Perceptron predicts -1, which corresponds to non-fiction .

# 2. How many total errors did the Perceptron make across all 10 epochs?
Sum of errors per epoch:  
`2 + 1 + 2 + 1 + 1 + 1 + 0 + 0 + 0 + 0 = 8`  
Total errors = 8

# 3. Why do the errors drop to 0 by epoch 7? What does this tell you about the dataset?
The Perceptron makes no mistakes after epoch 7 because it has learned the linear pattern in the dataset.  
This tells us that the dataset is linearly separable, meaning a straight line can perfectly separate fiction and non-fiction books based on size and color.

#  Excercise 2 :  Observations ,Questions and Answers

## Observations

#  How do the errors change over the 10 epochs?
Errors fluctuate initially but generally decrease, showing the Perceptron learning.

#  Which epoch does the librarian stop making mistakes?  
Errors reach 0 at epoch 7, meaning the Perceptron has learned to classify all books correctly.

## Questions and Answers

#  Why do the errors go up and down (e.g., 2, 1, 2, 1) before reaching 0?
Initial weights are random, so some predictions are correct while others are wrong.  
The Perceptron updates weights incrementally, causing fluctuations until convergence.

#  What does it mean when the errors reach 0?
The Perceptron has perfectly learned the sorting rule for the dataset.  
The dataset is linearly separable, so a straight line can correctly separate fiction and non-fiction books.

#  Excercise 3 :  Observations ,Questions and Answers

## Observations

#  Position of new book [3, 2]**:  
The new book falls in the red-shaded region, indicating it is classified as non-fiction (-1).

#  Decision boundary correctness**:  
The boundary separates fiction and non-fiction books correctly, with most blue circles on one side and red crosses on the other.

## Questions and Answers

#  Why does the new book [3, 2] get a prediction of -1?**  
The new book lies in the red region relative to the decision boundary, so it is classified as non-fiction (-1).

#  How does the decision boundary separate the fiction and non-fiction books?**  
The boundary is a linear line that divides the feature space such that books with smaller size and color are mostly non-fiction, and larger size/color are mostly fiction.

# If you test a new book at [4, 4], what prediction would you expect? Why? 
A new book at [4, 4] would fall in the blue region, so it is expected to be classified as fiction (+1) because it lies on the fiction side of the boundary.

#  Excercise 4 :  Questions and Answers

# How does changing eta (learning rate) affect the errors list? Does slower (0.01) or faster (0.5) learning make the errors drop faster?
Slower learning (eta=0.01): errors decrease gradually; takes longer to reach 0 (if it reaches 0).
Faster learning (eta=0.5): errors drop faster initially, but might overshoot or oscillate before settling.

# How does changing n_iter (epochs) affect the results? Did fewer epochs (n_iter=5) still reach 0 errors?
More epochs (20): gives the model more chances to correct mistakes; likely reaches 0 errors even with small eta.
Fewer epochs (5): may not reach 0 errors if learning is slow, but fast learning might already converge.

# Did the prediction for [3, 2] change with different settings? Why or why not? (Hint: Think about whether the final decision boundary changes.)
Didn’t change much.
Differences could happen if the decision boundary hasn’t fully stabilized with fewer epochs or small eta.

#  Excercise 4 :  Questions and Answers

# What does the prediction for [4.0, 1.0] mean? Is it Setosa (-1) or Versicolor (+1)?
Versicolor (+1)

# Does the errors list reach 0? Why or why not? (Hint: Research whether Setosa and Versicolor are linearly separable.)
Yes, because Setosa and Versicolor are linearly separable in petal length & width.
If classes were not separable (overlapping), errors would never reach 0.

# How does the decision boundary for the Iris data compare to the book dataset? Is it easier or harder to separate the classes?
Book dataset: boundary may be irregular due to small dataset & overlaps.
Iris dataset: boundary is clear and easier to separate.

#  Excercise 5 :  Questions and Answers
# Did the prediction for [3, 2] change after adding the new book? Why?
The prediction changed ,Since Perceptron adjusts the decision boundary when training, adding this positive point might shift the boundary upward, making [3, 2] more likely to be classified as 1.

# How did the errors list change? Did it still reach 0? (Hint: Is the new dataset still linearly separable?)
The errors list tracks how many misclassifications happen per epoch.
Yes, it can still reach 0 errors.

# Try changing random_state (e.g., 42 or 100). Does it affect the prediction or errors?
random_state just affects the initial random weights.
Changing random_state won’t stop convergence, but it might change how fast it converges and slightly shift the separating line.


