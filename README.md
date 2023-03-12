# nanoGPT
The simple repository for training/finetuning medium-sized GPTs.
<img src ="https://github.com/Pushkar1853/nanoGPT/blob/1460e488f1049b8b151408db495531b1852fc41a/images/model.png"  style: height="600px" width="auto" align="right" >
This is one attempt to build a version of training and finetuning nano-GPT model. So, the popular ChatGPT is for your information introduced below as mentionaed in the [OpenAI](https://openai.com/) website.

* The dialogue format makes it possible for [ChatGPT](https://openai.com/blog/chatgpt) to answer followup questions, admit its mistakes, challenge incorrect premises, and reject inappropriate requests.
* ChatGPT is a sibling model to [InstructGPT](https://openai.com/blog/instruction-following/), which is trained to follow an instruction in a prompt and provide a detailed response.

## What's the catch? 
This whole notebook is based on the vital research paper : 
[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

* The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.

## A message to the reader:

* The architecture of the model followed is a Multi-Head Attention, but with only "Self-Attention" Layers, without the use of Cross Attention from the encoder.
<img src ="https://github.com/Pushkar1853/nanoGPT/blob/1460e488f1049b8b151408db495531b1852fc41a/images/model2.png"  style: height="400px" width="auto" >
* The paper followed the task of translation, thereby requiring the process of encoder and decoder.
* Here, only the next predictions i.e. decoding layers are concerned, therefore, an extention of the Bigram Language Model from [makemore](https://github.com/Pushkar1853/makemore). Check out that as well.

## Methods
* We trained this model using Reinforcement Learning from Human Feedback (RLHF), using the same methods as InstructGPT, but with slight differences in the data collection setup. We trained an initial model using supervised fine-tuning: human AI trainers provided conversations in which they played both sides—the user and an AI assistant. We gave the trainers access to model-written suggestions to help them compose their responses. We mixed this new dialogue dataset with the InstructGPT dataset, which we transformed into a dialogue format.

* To create a reward model for reinforcement learning, we needed to collect comparison data, which consisted of two or more model responses ranked by quality. To collect this data, we took conversations that AI trainers had with the chatbot. We randomly selected a model-written message, sampled several alternative completions, and had AI trainers rank them. Using these reward models, we can fine-tune the model using Proximal Policy Optimization. We performed several iterations of this process.
<img src ="https://github.com/Pushkar1853/nanoGPT/blob/1460e488f1049b8b151408db495531b1852fc41a/images/ChatGPT_Diagram.svg"  style: height="600px" width="auto" align="right" >

## Install

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
- `pip install transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
- `pip install datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
- `pip install tiktoken` for OpenAI's fast BPE code <3
- `pip install wandb` for optional logging <3
- `pip install tqdm`

## Notebooks and codes:
```
notebooks/gpt-dev.ipynb
codes/bigram.py
codes/v2.py
```

## Dataset

```
data/input.txt
```

## Result Sample
This generates a few samples, for example:

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.
DUKE VINCENTIO:
I thank your eyes against it.
DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?
DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

## References

* [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/pdf/2210.10760.pdf)
* [Learning to summarize from human feedback](https://proceedings.neurips.cc/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf)



