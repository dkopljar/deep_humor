## How Deep is Your Humor 

### Authors:
* Bartol Freškura
* Filip Gulan
* Damir Kopljar

### Abstract
In this paper, we consider the task of comparative humor ranking in two manners: detecting which tweet of two is more humorous and
ranking the given tweets by how humorous they are in three classes. We opted for a different approach based on recent deep neural
models in order to eschew manual feature engineering. In evaluation section, we experimented with the bi-directional LSTM and CNN,
in combination and separately. For constructing feature vectors we used pre-trained Twitter GloVe word embeddings along with learned
character embedding. The system was trained, tuned, and evaluated on the SemEval-2017 Task 6 dataset for which it yields ambitious
results.

### Conclusion
We proposed three different models for solving comparative humor ranking tasks of pairwise comparison and direct
ranking classification. All three models use deep learning
architecture by combining approaches of recurrent and convolutional neural networks.
For pairwise comparison task best results were achieved
using the Bi-LSTM model result in 69.1% accuracy score
on unseen evaluation data, and for direct ranking classification task best results were achieved using same Bi-LSTM
model and were 0.881 on unseen evaluation data. Model
evaluation on final unseen data is done using official evaluation scripts given in SemEval-2017 Task 6.
We have compared our results with the results of other
task participants resulting in our model taking the first place
on the Task A, and ranking second on the Task B. The main
distinction between our model and competitive models is
the lack of hand engineered features which indicates that
automatic feature extraction using deep learning framework
has a great prospect in this task and requires further work.
For the next step, we would experiment with specially
adapted word embeddings trained only on the humor containing corpus. We believe it is crucial for word vectors
to learn semantic meaning from the domain specific data
because of the complex humor structure.
