import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from tqdm import tqdm
from datasets import Dataset
from .preprocess import pre_text_cleaning

def infere_model(model_path,input_data,device):
    
    ####################################################
    ## load the model and tokenizer from the checkpoint
    ####################################################
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    ########################################################################
    ## convert input dict data to data set format for preprocessing pipeline
    ########################################################################
    input_dataset = Dataset.from_dict({
    "example_id": input_data.keys(),
    "paragraph": input_data.values()
    })

    #########################################
    ## preprocessing pipeline for input data
    #########################################
    input_dataset_cleaned = (input_dataset
        .map(pre_text_cleaning, fn_kwargs={"col": "paragraph"}, batched=True, load_from_cache_file=False)
        # .map(segment_tokenize, fn_kwargs={"col": "paragraph", "by": "word"}, batched=True)
        # .map(remove_stopwords, fn_kwargs={"col": "paragraph", "by": "word"}, batched=True)
        # .map(qalsadi_lemma, fn_kwargs={"col": "paragraph", "by": "word"}, batched=True)
        # .map(translate, fn_kwargs={"col": "paragraph", "max_length": 1024}, batched=True)
    )

    #################################################################
    ## reconvert data to original format to start the generation step
    #################################################################
    example_id, paragraph = input_dataset_cleaned.to_dict().values()
    input_data = dict(zip(example_id, paragraph))

    results = {}

    #########################################
    ## starting text2text generation proecess
    #########################################
    for id in tqdm(input_data.keys(), desc="inferring model"):
        inputs = tokenizer(
            [input_data[id]],
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )

        tokens = len(tokenizer.encode(input_data[id]))
        print(tokens)

        outputs = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            num_beams=6,
            length_penalty=1.2,
            max_new_tokens=int(tokens * 0.4) - 20,
            min_new_tokens=20 if int(tokens * 0.3) < 20 else int(tokens * 0.3)               
        )

        summary = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        print(summary)

        results[id]=summary
  
    return results
