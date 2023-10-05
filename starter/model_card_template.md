# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model was built by me - Hau Pham. This model use Scikit-learn Random Forest Classifier and Census dataset.
## Intended Use

This model was built to predict salary of person.
## Training Data

The data use to train model is Census 1994 [here](https://github.com/udacity/nd0821-c3-starter-code/blob/master/starter/data/census.csv). This data was extracted from the [1994 Census bureau database](http://www.census.gov/en.html) by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics)

I use `train_test_split` function of Scikit-learn to split data into train, test dataset with ratio is 80-20.

The sample data:
```javascript
{
    'age': 39,
    'workclass': 'State-gov',
    'fnlgt': 77516,
    'education': 'Bachelors',
    'education-num': 13,
    'marital-status': 'Never-married',
    'occupation': 'Adm-clerical',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Male',
    'capital-gain': 2174,
    'capital-loss': 0,
    'hours-per-week': 40,
    'native-country': 'United-States',
    'salary': '<=50K'
}
```
## Evaluation Data

## Metrics
_Please include the metrics used and your model's performance on those metrics._

## Ethical Considerations

## Caveats and Recommendations
