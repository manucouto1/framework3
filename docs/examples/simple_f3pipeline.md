# Tutorial: F3Pipeline with Cache

This tutorial will guide you through a simple example of using F3Pipeline, including the Cache functionality.

## 1. Necessary Imports

First, we import the required classes and functions:

```python
from framework3 import F3Pipeline
from framework3.plugins.filters import StandardScalerPlugin
from framework3.plugins.filters import ClassifierSVMPlugin
from framework3.plugins.metrics import F1
from framework3.base import XYData
from framework3.plugins.filters import Cached
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from rich import print
```

## 2. Data Preparation

We load the Iris dataset and split it into training and test sets:

```python
# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create XYData objects
x_train = XYData.mock(X_train)
y_train = XYData.mock(y_train)
x_test = XYData.mock(X_test)
y_test = XYData.mock(y_test)
```

## 3. Creating the Pipeline with Cache

We create a pipeline that includes a cached `StandardScalerPlugin` and a `ClassifierSVMPlugin`:

```python
pipeline = F3Pipeline(
    filters=[
        Cached(
            filter=StandardScalerPlugin(),
            cache_data=True,
            cache_filter=True,
            overwrite=False
        ),
        ClassifierSVMPlugin()
    ],
    metrics=[F1()]
)
print(pipeline)
```
```bash
F3Pipeline(
    filters=[
        Cached(filter=StandardScalerPlugin(), cache_data=True, cache_filter=True, overwrite=False, storage=None),
        ClassifierSVMPlugin(C=1.0, kernel='linear', gamma='scale')
    ],
    metrics=[F1(average='weighted')],
    overwrite=False,
    store=False,
    log=False
)
```

## 4. Training the Model

We train the model with the training data:

```python
pipeline.fit(x_train, y_train)
```
```bash
Fitting pipeline...
****************************************************************************************************
* Cached({'filter': StandardScalerPlugin({}), 'cache_data': True, 'cache_filter': True, 'overwrite': False, 'storage': None}):
        - El filtro StandardScalerPlugin({}) con hash 982eff29324db9dcac5b7f04df712160af35eb23 No existe, se va a
entrenar.
         - El filtro StandardScalerPlugin({}) Se cachea.
	 * Saving in local path: cache/StandardScalerPlugin/982eff29324db9dcac5b7f04df712160af35eb23/model
	 * Saved !
         - El dato XYData(_hash='74a1e6bc3696616ef498c7303636d4581d9c386e', _path='StandardScalerPlugin/982eff29324db9dcac5b7f04df712160af35eb23') No existe, se va a crear.
         - El dato XYData(_hash='74a1e6bc3696616ef498c7303636d4581d9c386e', _path='StandardScalerPlugin/982eff29324db9dcac5b7f04df712160af35eb23') Se cachea.
	 * Saving in local path: cache/StandardScalerPlugin/982eff29324db9dcac5b7f04df712160af35eb23/74a1e6bc3696616ef498c7303636d4581d9c386e
	 * Saved !
* ClassifierSVMPlugin({'C': 1.0, 'kernel': 'linear', 'gamma': 'scale'}):
```

## 5. Prediction and Evaluation

We make predictions and evaluate the model:

```python
# Make predictions
predictions = pipeline.predict(x_test)

# Evaluate the model
evaluation = pipeline.evaluate(x_test, y_test, predictions)
print("Evaluation results:", evaluation)
```
```bash
____________________________________________________________________________________________________
Predicting pipeline...
****************************************************************************************************
* Cached({'filter': StandardScalerPlugin({}), 'cache_data': True, 'cache_filter': True, 'overwrite': False, 'storage': None})
         - El dato XYData(_hash='4712400607293098af5ca698ea46c0ff9b7818f9', _path='StandardScalerPlugin/982eff29324db9dcac5b7f04df712160af35eb23') No existe, se va a crear.
         - El dato XYData(_hash='4712400607293098af5ca698ea46c0ff9b7818f9', _path='StandardScalerPlugin/982eff29324db9dcac5b7f04df712160af35eb23') Se cachea.
	 * Saving in local path: cache/StandardScalerPlugin/982eff29324db9dcac5b7f04df712160af35eb23/4712400607293098af5ca698ea46c0ff9b7818f9
	 * Saved !
* ClassifierSVMPlugin({'C': 1.0, 'kernel': 'linear', 'gamma': 'scale'})
____________________________________________________________________________________________________
Evaluating pipeline...
____________________________________________________________________________________________________
Resultados de la evaluación:
{'F1': 0.9664109121909632}
```

## 6. Demonstrating Cache Usage

We run the pipeline again to demonstrate the use of cache:

```python
print("Second execution (should use cached data):")
pipeline.fit(x_train, y_train)
predictions = pipeline.predict(x_test)
evaluation = pipeline.evaluate(x_test, y_test, predictions)
print("Evaluation results:", evaluation)
```
```bash
Segunda ejecución (debería usar datos en caché):
____________________________________________________________________________________________________
Fitting pipeline...
****************************************************************************************************
* Cached({'filter': StandardScalerPlugin({}), 'cache_data': True, 'cache_filter': True, 'overwrite': False,
'storage': None}):
         - El filtro StandardScalerPlugin({}) Existe, se crea lambda.
         - El dato XYData(_hash='74a1e6bc3696616ef498c7303636d4581d9c386e', _path='StandardScalerPlugin/982eff29324db9dcac5b7f04df712160af35eb23') Existe, se crea lambda.
* ClassifierSVMPlugin({'C': 1.0, 'kernel': 'linear', 'gamma': 'scale'}):
	 * Downloading: <_io.BufferedReader name='cache/StandardScalerPlugin/982eff29324db9dcac5b7f04df712160af35eb23/74a1e6bc3696616ef498c7303636d4581d9c386e'>
	 * Downloading: <_io.BufferedReader name='cache/StandardScalerPlugin/982eff29324db9dcac5b7f04df712160af35eb23/74a1e6bc3696616ef498c7303636d4581d9c386e'>
____________________________________________________________________________________________________
Predicting pipeline...
****************************************************************************************************
* Cached({'filter': StandardScalerPlugin({}), 'cache_data': True, 'cache_filter': True, 'overwrite': False,
'storage': None})
         - El dato XYData(_hash='4712400607293098af5ca698ea46c0ff9b7818f9', _path='StandardScalerPlugin/982eff29324db9dcac5b7f04df712160af35eb23') Existe, se crea lambda.
* ClassifierSVMPlugin({'C': 1.0, 'kernel': 'linear', 'gamma': 'scale'})
	 * Downloading: <_io.BufferedReader name='cache/StandardScalerPlugin/982eff29324db9dcac5b7f04df712160af35eb23/4712400607293098af5ca698ea46c0ff9b7818f9'>
____________________________________________________________________________________________________
Evaluating pipeline...
____________________________________________________________________________________________________
Resultados de la evaluación:
{'F1': 0.9664109121909632}
```

In this second execution, you should notice that the `StandardScalerPlugin` uses the cached data, which may result in faster execution time.

## Conclusion

This tutorial has demonstrated how to use F3Pipeline with the Cache functionality in Framework3. We've seen how to create a pipeline that includes a cached filter, how to train the model, make predictions, and evaluate performance. We've also shown how caching can improve performance in subsequent runs.
