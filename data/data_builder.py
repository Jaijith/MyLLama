import os
import struct
import random

import numpy as np
from data.data_config import dtypes, HDR_MAGIC, HDR_SIZE, code


class PackedDatasetBuilder(object):
    """
    __init__: this is the contruction of the class. It intializes the class with several paearms like 
    output directory, prefix for the file names, chunk size, a seperation token, data tyeps, and 
    vocabulary size. If data type is "auto", it will set the data type based on the vocab size. Then it
    creates an empty numpy array of chunk size with sepecified data types and fills it with the speration token
    
    _write_chunk : This is a private method which writes the current numpy to a binary file.
    The filename is created using prefix and counter. The method first wirtes some header information
    (HDR_MAGIC, version, data type code, chunk size) to the binary file, then the numpy arry is written as bytes.
    After writing , it increments the counter, refills the arry with the seperation token and resets the index
    
    dtype: property which returns the current data type of the array
    
    filenames: property which returns a copy of the list of filenames that have been written
    
    add_array: the methos adds a new numpy arry to th existing one. if adding new arry would exceed the 
    chunk size, it writes the filled part to a binary file and then continues with the reminder of the new arrya
    
    write_reminder: this method is used to write any remaining adata in the arry to a binary file. it can be
    used adter all arrays have beena dded.
    """
    def __init__(
        self,
        outdir,
        prefix,
        chunk_size,
        sep_token,
        dtype="auto",
        vocab_size=None,
    ):
        if dtype == "auto":
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            if vocab_size is not None and vocab_size < 65500:
                self._dtype = np.uint16
            else:
                self._dtype = np.int32
        else:
            self._dtype = dtype
        self._counter = 0
        self._chunk_size = chunk_size
        self._outdir = outdir
        self._prefix = prefix
        self._sep_token = sep_token
        self._arr = np.zeros(self._chunk_size, dtype=self._dtype)
        self._arr.fill(self._sep_token)
        self._idx = 0
        self._version = 1
        self._filenames = []

    def _write_chunk(self):
        filename = f"{self._prefix}_{self._counter:010d}.bin"
        filename = os.path.join(self._outdir, filename)

        with open(filename, "wb") as f:
            f.write(HDR_MAGIC)
            f.write(struct.pack("<Q", self._version))
            f.write(struct.pack("<B", code(self._dtype)))
            f.write(struct.pack("<Q", self._chunk_size))
            f.write(self._arr.tobytes(order="C"))

        self._filenames.append(filename)
        self._counter += 1
        self._arr.fill(self._sep_token)
        self._idx = 0

    @property
    def dtype(self):
        return self._dtype

    @property
    def filenames(self):
        return self._filenames.copy()

    def add_array(self, arr):
        while self._idx + arr.shape[0] > self._chunk_size:
            part_len = self._chunk_size - self._idx
            self._arr[self._idx : self._idx + part_len] = arr[:part_len]
            self._write_chunk()
            arr = arr[part_len:]

        arr_len = arr.shape[0]
        self._arr[self._idx : self._idx + arr_len] = arr
        self._idx += arr_len

    def write_reminder(self):
        self._write_chunk()
