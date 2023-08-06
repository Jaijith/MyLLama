import os
import tqdm
import json
import numpy as np
from pathlib import Path
from data.data_config import filenames_sample
from data.data_builder import PackedDatasetBuilder

def prepare_data(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int, 
) -> None:
    #What is chunk size
    ### Prepare the dataset assuming tokenization exists
    
    #create the destination path if doesnt exis
    destination_path.mkdir(parents=True, exist_ok=True)
    #get the tokenizer.assumed here that it exists
    tokenizer = "need to create one"
    
    #for each file in the list, run the full pipelines
    for name in filenames_sample:
        
        #create filepath
        filepath = source_path / name
        if not filepath.is_file():
            raise RuntimeError(f"Input file not found at {filepath}. \n")
        
        #get the root and ext 
        #for e.g for path "/home/User/Desktop/file.txt"
        #it splits to /home/User/Desktop/file  as root
        #and .txt as extention
        prefix, _ = os.path.splitext(name)
        
        builder = PackedDatasetBuilder(
            outdir=destination_path,
            prefix=prefix,
            chunk_size=chunk_size,
            sep_token=tokenizer.bos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )

        print(f"Processing {name}")

        with open(filepath, encoding="utf-8") as f:
            for row in tqdm(f):
                text = json.loads(row)["text"]
                text_ids = tokenizer.encode(text)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))

        builder.write_reminder()
        
        
        
            
    
    