## finetuning model
#### selection of parameters
![selection of parameters](https://github.com/mkgs210/new-adapter-project/assets/78417431/0a8da03e-5b33-44f0-a7a6-d5bccbcfdb89)
#### training visualization
![training visualization](https://github.com/mkgs210/new-adapter-project/assets/78417431/b9b72fc8-f21b-4c98-803f-fe08f96ae87f)

## model inference
#### interaction with the model
![interaction with the model](https://github.com/mkgs210/new-adapter-project/assets/78417431/f1bf7e70-f9f6-4629-ab9c-47bd9db862c6)


## start service
```
pip install requirements.txt
python start.py
```

## Task List
- [ ] finetuning tasks
    - [x] text classification task
    - [x] text generation task
    - [ ] token classification task
    - [ ] text summarization task
    - [ ] masked language modeling task
    - [ ] question answering task
- [x] adapters
    - [x] Pfeiffer Bottleneck
    - [x] Houlsby Bottleneck
    - [x] Parallel Bottleneck
    - [x] Scaled Parallel Bottleneck
    - [x] Compacter
    - [x] Compacter++
    - [x] Prefix Tuning
    - [x] Flat Prefix Tuning
    - [x] LoRA
    - [x] (IA)^3
    - [x] MAM Adapter
    - [x] UniPELT
- [ ] adapters compositions
    - [ ] stack
    - [ ] fuse
    - [ ] split
    - [ ] parallel
- [x] methods and tools for efficient training
    - [x] gradient checkpointing
    - [x] mixed precision training
    - [x] quantization
    - [x] optimizer choice
