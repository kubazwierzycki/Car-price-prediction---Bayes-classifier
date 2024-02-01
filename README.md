# Car price prediction - Bayes classifier

The purpose of the program is to test the efficiency of the Naive Bayes classifier when
the problem is to predict the price range of the car. <p>
The training and testing data source is "100,000 UK Used Car Data set" dataset from Kaggle.

## How it works?

The program efficiency is tested at various widths of the vehicle's price range.
Up to them, the training data is aggregated into proper price ranges
(the lower the price range, the more different ranges with less data for each of them)
which is used to calculate the probability of the tested car's price. <p>
The final prediction is based on the three most probable price ranges.

## What could be done better?

- it would be good to measure the execution time as well
- there are some numbers in the code that should be replaced with constant values
- each function is described in comments, but there is a lack of parameter description
