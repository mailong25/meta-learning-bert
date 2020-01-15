# Fewshot text classification with meta learning and BERT
## Requirements
  - transformers==2.2.1
  - python>=3.6
  - torch==1.3.0

# Introduction
This repository is an implementation of First-order MAML and Reptile on top of BERT. For those who interested in Second-order MAML, we also provide a script that performs functional forward with specified BERT's weights

Application can be use for domain adaptation with limited training examples. In our case, we try to build an build an accurate sentiment analysis model of Amazon product reviews for low-resource domains (~ 80 training examples/domain).

What is domain adaptation?

Domain adaptation is a field associated with machine learning and transfer learning. This scenario arises when we aim at learning from high-resource domains a well performing model on low-resource (but related) domains.

For example, this is the number of training examples per domain in our case:

'apparel': 1717,

'baby': 1107,

'beauty': 993,

'books': 921,

'camera_&_photo': 1086,

'cell_phones_&_service': 698,

'dvd': 893,

'electronics': 1277,

'grocery': 1100,

'health_&_personal_care': 1429,

'jewelry_&_watches': 1086,

'kitchen_&_housewares': 1390,

'magazines': 1133,

'music': 1007,

'outdoor_living': 980,

'software': 1029,

'sports_&_outdoors': 1336,

'toys_&_games': 1363,

'video': 1010,

'automotive': 100,

'computer_&_video_games': 100,

'office_products': 100
         
According to the statistics, We have 100 training examples for "office_products", "automotive", "computer_&videogames". Can we still build an accurate model on these domain ? 

-> Absolutetly, we actually achieved an average accuracy of 93% on test domains with just 80 training examples.

# Solution 
We leverage data from high-resource domains to create a good "starting point". From this point, we start training a specific model for low-resource domain.

## Approach 1: Transfer learning
We train a single model (Model_X) on concatenated data from high-resource domains. Then, we retrain Model_X on low-resource domain

## Approach 2: Meta learning
We stimulate a lot of situations where the Model_X are forced to learn fast with limited training datad. The model_X are getting better at "learning with less" after each training situation. We called these situations as Meta-task. Each task contain two sets:
 - Support set: contain few training samples
 - Query set: Provide learning feedback. The model use this feedback to adapt its learning strategy
 
There meta tasks is constructed from high-resource domains, serving as meta training data.

So, what is the form of "learning strategy" of a learner ? . It's simply an initialization of weights.
 - A good learner (learner that learn fast and obtrain good result on test-set) have a good initialization of weight, which can be easily tunned on data from new domains.
 - A bad learner simply have a bad initialization of weights.

In other words, meta training is simply a process of learning to initialize model's weights such that these weights can be easily tunned. 

# Example usage
Please take a look at Interactive.ipynb
