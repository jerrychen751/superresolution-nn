This covers how to add a new model into our repository.

### Overview
1. Write the model file containing the model architecture, how to make the training pair, and the custom dataset class (different models will require different representations of training data).
2. Depending on what data the model needs, write a new download + preprocess config YAML file if needed. Within `preprocess.py`, we take downloaded training data and preprocess it, saving it to configured filepaths. Also adjust preprocess.py to read the proper config/model to perform correct preprocessing.
3. 