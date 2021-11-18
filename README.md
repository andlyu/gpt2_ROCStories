# gpt2_ROCStories

# Summary

This repo provides a setup with which to experiment with text generation for ROC Stories using GPT2.

ROC Stories is a dataset of 5 sentence stories, where the first 4 sentences are used as input, and the last one is predicted.

On top of that dataset, the characters were masked using <MALE>, <FEMALE>, and <NEUTRAL> masks in order to let the language model focus more on the plot of the story instead of character names. While the text generation is repetative this part can still be improved upon when using the gpt2 model.

The following code will donwload and run the fintuning step and evaluation step, where we evaluate the repetativeness of the sentences. 

```bash
git clone https://github.com/andlyu/gpt2_ROCStories.git
cd gpt2_ROCStories/
bash all_setup.sh
conda activate roc_env 
bash run_finetune.sh #Finetunes GPT2 model on the dataset
bash run_eval.sh     #Evaluates the output of the finetuned model (Example outputs provided)
```
