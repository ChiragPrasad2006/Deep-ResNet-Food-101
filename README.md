# Deep-ResNet-Food-101
A deep CNN with ResNet architecture built from scratch for 101 Types of Foods

url: https://huggingface.co/Kami0867/resnet-101-food101/commit/c017074ad3b50bec933e8eff403fcedd054ec32a

# Loading Model

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification

model_id = "Kami0867/resnet-101-food101"

image_processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)
```

# Dataset 

dataset: kaggle.com/datasets/kmader/food41?select=images

# Achieved accuracy 
train-set: 94.28%

test-set: 84.86%
