# Data Classification Learning Algorithm

## About

This project utilizes a simple AI learning algorithm to classify ad data based on most discriminating and least discriminating terms.

## Versioning

**VERSION:** 1.0

**RELEASE:** N/A

**LAST UPDATED:** April 19th, 2019

## Resources

**/data/:** Contains data for both training the learning algorithm and testing it.

## How To Use

After cloning the repository, run
```bash
python classify.py
```
with the following arguments:

**Argument 1** must be a set of testing data and **Argument 2** must be a set of traning data for the learning algorithm.

**Argument 3** must be a valid method to run. The options are:

- 'tf' - Creates a .csv file containing term frequencies for words in the provided data.

- 'tfgrep' - Finds the most discriminating term in the data a uses it to generate a confusion matrix for the data.

- 'priors' - Finds the most probable class in the training data and uses the results to generate a confusion matrix for the data.

- 'mnb' - Predicts the most likely class given a document using the data from tf.

- 'df' - Creates a .csv file containing document frequencies for words in the provided data.

- 'nb' - Predicts the most likely class given a document using the data from df.

- 'mine' - Uses a modified version of the tf data to predict the most likely class.

## Future Development

There are no plans for future development at this time.
