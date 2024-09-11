## The vital question is the access of remote HuggingFace datasets and models.
To solve this problem, I use model and datasets locally.

Download the model and datasets in [Huggingface mirror](https://hf-mirror.com)

More details cite to [Huggingface mirror](https://hf-mirror.com).
```
# install huggingface-cli
pip install -U huggingface_hub

# change enviroment variable
export HF_ENDPOINT=https://hf-mirror.com

# download model
huggingface-cli download --resume-download gpt2 --local-dir gpt2

# download dataset
huggingface-cli download --repo-type dataset --resume-download wikitext --local-dir wikitext
```
