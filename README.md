# file description

`label_classification.py`: Transform the output length into labels and predict what label the output length belongs to.
`rank.py`: Transform the output length into scores, longer output has higher score.
`generate_vectors.py`: Generate activations and store them as `.pt` file.
`length_gen_vllm.py`: Generate response, get response length.

# commands to run experiments

activation generation:
`python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --use_base_model --data_path ./datasets/alpaca_data.json --dataset_name alpaca`

inference using vllm and get the response length:
`python length_gen_vllm`

train a length predictor:
`python label_classification.py` or `python rank.py`
(currently I use activations from all layers for the prediction, you may also specify which layers you want to get activations from by changing `line 12-16, line 37` in `label_classification.py`.)

# tips

You may have to change the data/model path based on your own experiment environment
You may export your huggingface API key: `export HF_TOKEN="hf_Q...."`
