# handwrittenEsposallesCTC
Handwritten recognition model for Esposalles datasets, based on BLSTM and CTC.

## My software environment:

- Ubuntu 16.04 x64
- Python 3.5
- TensorFlow 1.1

## Structure:

It is a handwritten recognition model based on BLSTM and CTC. At this moment, I just use the simplest way to implement it: only 1 convolutional layer for feature extraction and followed by 1 BLSTM layer, and CTC as the loss function.

## Esposalles Datasets:

In Esposalles datasets (available at [Esposalles Datasets](http://rrc.cvc.uab.es/?ch=10&com=introduction)), there are 2 types: textline-based and word-based.

![](https://user-images.githubusercontent.com/9562709/29869617-047d88f0-8d84-11e7-8fb0-3bcf56b83cbf.png)

##### Figure 1. Textline-based Esposalles datasets

![](https://user-images.githubusercontent.com/9562709/29869636-1781cac4-8d84-11e7-8535-3591b9106930.png)

##### Figure 2. Word-based Esposalles datasets

Note:
In my repository there is a folder named "groundTruth", which is to make the datasets easier using. So when you download the Esposalles datasets, please copy the groundtruth txt files to the corresponding folders.

## Usage:

- **esposallesData.py**  is to preprocess the Esposalles datasets, you can change to textline-based datasets or word-based ones by changing the bool value "TEXTLINE".
- **esposallesSequenceCTC.py**  is the main program which has a class of SeqLearn(). 

## Result:

During the running of the program, "train_cer.log" and "test_cer.log" will generate. When it finished, the character error rate can be visualized by **showplt.py**. For textline-based datasets (with the batch size of 8), the test CER reaches 10% at around 400th epoch. For word-based datasets (with the batch size of 64), the test CER reaches 13.5% at around 400th epoch. Here is the demo result:

![](https://user-images.githubusercontent.com/9562709/30711883-6c4d2be2-9f0a-11e7-873d-c54bf4c150bb.png)

##### Figure 3. Character error rate for textline-based datasets

![](https://user-images.githubusercontent.com/9562709/30866931-dfc60fde-a2da-11e7-8c53-6df72f0dd944.png)
##### Figure 4. Character error rate for word-based datasets

## To be improved:

+ ~~This model has only 2 layers: one is convolutional and the other is BLSTM, so if more layers added, the result will be much better.~~ 
+ Dynamic learning rate
