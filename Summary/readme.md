# Summary deepnet with Tensorflow

play code for a "summary" net that will create a summary of a text.
Training and test examples are available in /data directory.

The net is based on the paper "Attention is All You Need".

The net has been trained with 100 epochs and the weights are saved in /weights directory.
execute test_pipeline.py to test some summaries from the training and test set.

## conclusion
The net is clearly overfitting the training data and is failing to generalise for the test set.
Typical solutions:
- Work with more data
- Add regularisation (there is a dropout layer to play with and perhaps some L2 regularisation could be added)
- change some of the hyper-parameters