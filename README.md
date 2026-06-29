# CustomLM-SFT
A custom generative Large Language model trained with pure Pytorch with code tutorial for two major phases in GenAI training: Pretraining and Supervised FineTuning. Our goal here is to remove as much noise as possible from standard implementations you find elsewhere and focus on teh core-training with minimal readable code. 

[Note: `Large` above is subjective but in today's terms its definitely not considered big enough. But as long as we are successfully able to train >150M model that is close to human-level coherence, we should be good :) ]

The underlying model trained was an autoregressive GPT-styled decoder-only model with a mix of Mixture of Experts and Dense Layers. To train a small 100-200M model like outs, the code repository fixed the `n_experts=16` and `top_k=2`and `lb_loss=1e-3` in MoE layers which means during training we only activated 2 best experts per token out of total of 16 with a small 0.1% weightage in load balancing loss in the objective.  

### Installation and Setup
The code-base was developed and tested on Linux systems with GPU, although you could make config changes and should pretty well on Windows 11 for SFT stage.
You can start by setting up the python environment and module installations. To run pretraining, PyTorch version should not matter much as long as you have PyTorch2.x
```bash
cd CustomLM-SFT
python3 -m pip install --upgrade pip
sudo apt install python3-venv
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
#if Torch installation does not work:
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
chmod +x hf_exports.sh
source ./hf_exports.sh
```

### Pretraining Data Prep

The prep code is loosely based on Karpathy's nanoGPT repo, but watered-down to make it look simplified. Once the the prep sripts complete the run, you end up with `train.bin` and `val.bin` files which carry the logic of encoding the natural language, tokenizing and then saving as one big Bin file. The tokenization stage uses a GPT-2 tokenizer through OpenAI's tiktoken library. Following datasets were tested and used for this pretraining exercise: `bookcorpus`, `OpenWebText`, `fineweb-edu`(Educational content subset of FineWeb dataset) and `RedPajama`(Github, wikipedia and stackexchange)
Make sure you have atleast 100G Diskspace and a minimum of 128G RAM before venturing into preparation of the pretraining datasets. You could also adjust the `num_workers` according to your HW availability.

### Pretraining
The objective of pretraining is simply the next token prediction, so we slice 0 to n-1 from the input and 1 to n from the same inputs and train the model with a CE loss objective(+ Load_balancing_loss). There is no "trick" here in the recipe, the model learns internal representations relating to the language such as avoiding trivial grammatical errors, reproducing syntax, understanding sentence formations etc. We should not expect the model to learn anything other than this, like factuality and accuracy of writing essays on given topics, which lie outside of the distribution learned by the model's weights.

```bash
python prepare_<dataset_name>.py
# for OpenWebText, just use:
python prepare.py
```

Once the dataset is prepared:
- Run the `train_moe.ipynb` Notebook
- Make sure to install the `accelerate` library-> This is the one of only two parts where we rely on HF libraries since it abstracts much of the PT's DDP framework. Our focus here is minimize focus on Distributed computing and rather focus more on the core LLM training   
- `num_processes` depends on the num of CUDA devices on your system
- Additional MoE training configurations can be fine-tuned in `moe_config.json` before running teh notebook

### SFT
In the supervised fine-tuning stage, we essentially perform the following with a couple of tricks in the recipe:
- Prepare the datamix by loading the datasets with QA pairs and pass through a chat template with tags like `### INSTRUCTION` and `### RESPONSE`
- In each QA pair, the question corresponds to input prompts and answers correspond to the labels
- The first trick is:
      - For Q's we take full token sequence (i.e., corresponding to the prompt + response + EOT)
      - For A's, we mask out the tokens corresponding to the prompt, and append it to the token_id for response tokens
- The second trick is to use a "System Prompt" in the chat template after the `### INPUT` depending on the dataset sample in the mix which reflects what the model is supposed to expect in the question. This is the key to going the "extra mile" since we are dealing with a small model with small enough context length.
- Tokenize the dataset object with the same tokenizer config as used in the pretraining stage  
- We then train the "shifted token" CE loss objective (+ Load_balancing_loss) on the pretrained model with a low enough lr

To prepare the SFT datamix, you can configure the data mix configurations in `sft_config.json` and then run:
```bash
python sft_prepare.py --config sft_config.json
```

Once the run is complete, you should see the data in: `./sft_data`

Then run the `sft_train.ipynb` similar to the pretraining notebook by adjusting the hyperparams in `training_config` and `checkpoint_config` in sft_config.json. However, keep in mind the chinchilla limit and SFT passes should be much less than this or you could encounter the catastrophic forgetting problem. 

#### Visualization (Optional)
You can use the `visualize_att.ipynb ` notebook that utilizes a very handy module called `bertviz` which can be installed with pip. It helps us visualize each transformer layer using a drop-down or for each head in the attention layer (color-coded) and shown below:

<img width="600" height="557" alt="image" src="https://github.com/user-attachments/assets/3fab5604-f181-4b06-8cb8-b371c67844d5" />

This helps you validate what tokens the model learned to attend to during training.

### Inference

For a single prompt, run 

```bash
python sft_generate_bkp.py ./sft_checkpoints/sft_ckpt_final.pt \\
      --instruction "Write a small poem about rain." \\
      --temperature 0.9 --top-p 0.95 --top-k 50 --max-tokens 200
```

If you want to run it on a validation held-out, you can run:

```bash
python sft_generate_bkp.py ./sft_checkpoints/sft_ckpt_final.pt \\
      --eval-source gsm8k --eval-file ./sft_data/val.jsonl \\
# Optional sampling params that you can use: --temperature 0.8 --top-p 0.95 --top-k 50 --max-tokens 200 --repetition-penalty 1.1      
```

### Next Steps

- Converting the model into a safetensors format for HF repos
- Post-training (Phase 3): Performing RLHF/DPO on the fine-tuned model 
