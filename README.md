# Deep-ResNet-Food-101
A deep CNN with ResNet architecture built from scratch for 101 Types of Foods

url: https://huggingface.co/Kami0867/resnet-101-food101/commit/c017074ad3b50bec933e8eff403fcedd054ec32a

# Loading Model

**model_id = "Kami0867/resnet-101-food101"**

**tokenizer = AutoTokenizer.from_pretrained(model_id)**

**model = AutoModelForSequenceClassification.from_pretrained(model_id)**

# Dataset 

dataset: kaggle.com/datasets/kmader/food41?select=images

# Achieved accuracy 
train-set: 94.28%

test-set: 84.86%
