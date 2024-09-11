# Sandbox for PyTorch Lightning

PyTorch Lightning has out of the box sharding support. 

- Can we leverage this for distributed multi GPU experiments for training and inference? 
- Will sharding work for small models, which don't necessarily need sharding for inference, but can be trained with Distributed Data Parallel?

Note, this is simply a sandbox for @sanjif-shanmugavelu, **not** serious code whatsoever.