# TLGA

TLGA is an abstractive model proposed for Financial News Summarization, which contains three major modules. The Transformer-BiLSTM encoder can only learn long-range dependencies but also capture sequential-aware contextual information. The Graph Attention-based decoder can fully utilize the historical information of decoded tokens and capture key causal relations. Moreover, the history-enhanced attention can concentrate on salient input content based on history semantics, guiding our decoder to generate the summary around the corresponding contents.



# Code

## Environment

* python  3.7
* jieba  0.42.1
* torch  1.8.0
* tensorboard   1.14.0
* tensorflow-gpu  1.14.0
* pyrouge   0.1.3

## Usage

1. How to train the model
   
   ```
   python train.py 
   python train.py -m /your/model_path/    # Path to the saved model. If empty, it will start a new training. 
   ```

2. How to test the model
   
   ```
   python decode.py /your_model_path/   # where you save the model
   ```

## Evaluation Metrics

### Intrinsic Evaluation
* ROUGE (Recall-Oriented Understudy for Gisting Evaluation) 

   ROUGE measures the overlap between the system-generated summary and one or more reference summaries. The official pakage ROUGE-1.5.5 is used to evaluate this model, and we report the F1 score of Rouge-1, Rouge-2, and Rouge-L.
* BART_Score

   To assess faithfulness and informativeness. It computes the weighted log probability when using the fin-tuned financial BART baseline to convert the predicted text to/from a reference output or the source text.
### Extrinsic Evaluation

* We conduct a stock movement prediction task to evaluate the causality and informativeness of predicted summaries. We only employ generated summaries (no extra features) to decide on selling or buying shares of listed companies. Accuracy, F1-score, and Matthews Correlation Coefficient (MCC) are considered metrics for evaluation.

# Dataset

We have crawled 498,209 news articles and their headlines from several major financial portals from January 2013 to June 2020, including East Money, Sina Finance, China Business News, Securities Times, and Shanghai Securities News. Using the regular expression matching method, we remove extraneous noise from the raw texts (i.e., special symbols, URL links, and stock codes) and obtain 430,820 article-title pairs. We split them into a training set (420,820), a validation set (5,000), and a testing set (5,000).

The raw texts and preprocessed data will be publicly available in the near future. If you want to use this dataset, please request my supervisor at qkpeng@xjtu.edu.cn.


