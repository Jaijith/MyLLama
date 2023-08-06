import numpy as np

filenames_sample = [
    "arxiv_sample.jsonl",
    "book_sample.jsonl",
]

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}

#HDR_MAGIC is a magic string used to identify files in the expected format
#HDR_SIZE is the size of the header in each file.

HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes

# Data Proportion ftom https://arxiv.org/pdf/2302.13971.pdf 
#This proportion is used to create the dataset for training purposses

data_config = [
    ("arxiv", 2.5),
    ("book", 4.5),
    ("c4", 15.0),
    ("cc", 67.0),
    ("github", 4.5),
    ("stackexchange", 2.0),
    ("wikipedia", 4.5),
]

def code(dtype):
    """
    Returns the code corresponding to a numpy data type.
    Args:
        dtype: A numpy data type.
    Returns:
        The code corresponding to the data type.
    Raises:
        ValueError: If the data type is not in the dtypes dictionary.
    """
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)

