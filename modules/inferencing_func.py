import pandas as pd
from PIL import Image

import torch
import re
import io

DEVICE = "cuda: 0" if torch.cuda.is_available() else "cpu"

def json_to_df(data):
    print(data)

    dfs = []
    for key, value in data.items():
        dfs.append((
            pd.json_normalize(value, sep="_")
            .rename(columns=lambda x: x.replace(".", "_")), key
        ))

    return dfs


def inference_fn(invoice: str, model, processor):
    # Reading image
    if type(invoice) == str:
        img = Image.open(invoice)
    
    else: 
        img = invoice 

    # Convert to RGB
    img = invoice.convert("RGB")

    # Prompting task
    task_prompt = "<s_cord-v2>"

    # Tokenization
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    # Pixel Values
    pixel_values = processor(img, return_tensors="pt")['pixel_values']

    # Generating Outputs
    outputs = model.generate(
        pixel_values.to(DEVICE),
        decoder_input_ids=decoder_input_ids.to(DEVICE),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )


    # Decoding Sequence
    sequence = processor.batch_decode(outputs.sequences)[0]

    # Preprocessing sequences
    sequence = (
        sequence
        .replace(processor.tokenizer.eos_token, "")
        .replace(processor.tokenizer.pad_token, "")
    )

    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

    # Decoding Tokens
    json_data = processor.token2json(sequence)

    # converting into dataframe
    return json_to_df(json_data)
