# quora_questions
baseline submission for quora question pairs kaggle challenge.  

This is a simple LSTM-based approach.  It didn't fare too well on the leaderboard.  

I think the main problems are:

1) Limited coverage of word vectors
2) Poor grammar of the question/answer pairs

Given more time, I would look into appending the glove vectors with character vectors to better deal with oov words.  
Generally though, I was hoping to address this with massive transfer learning by pre-training an auto-regressive LSTM 
on some large corpus.  Due to the poor grammar of the questions, however, I don't think that this approach would do that well.

The dataset does provide a good test for certain real-world settings where language doesn't come with consistant syntax.  
