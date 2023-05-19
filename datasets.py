import torch
import os
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm

DATA_FOLDER = "data/fma/spectrograms/"
GENRES_PATH = "data/fma/included_genres.txt"
SAMPLE_RATE = 44100

class FMA(torch.utils.data.Dataset):
    def __init__(self, workers=10, sample_shape=(256, 512)):
        pool = ThreadPool(workers)
        self.data = []
        self.sample_rate = SAMPLE_RATE
        self.mmap_samples = None
        self.mmap_path = "data/fma/memmap.dat"
        self.size = 0

        # Collect specified genres:
        with open(GENRES_PATH, "r") as file:
            lines = file.readlines()
            genres = [line.strip() for line in lines]
        
        # Determine size of dataset:
        for result in pool.map(self.count_files, genres):
            self.size += result
        
        pool = ThreadPool(workers)

        # Initialize memmap:
        print("INFO: Reading memmap")
        mode = "r+" if os.path.exists(self.mmap_path) else "w+"
        self.mmap_samples = np.memmap(
            filename=self.mmap_path,
            mode=mode,
            dtype=np.float32,
            shape=(self.size, sample_shape[0], sample_shape[1])
        )

        # Collect songs in each genre:
        if mode in "w+":
            for result in tqdm(pool.map(self.get_files, genres), total=len(genres), desc="Genre"):
                length = len(result)
                self.mmap_samples[0:length][:] = result[:]

    def __getitem__(self, idx):
        return self.mmap_samples[idx]
    
    def __len__(self):
        return len(self.mmap_samples)

    # Return the number of files per genre:
    def count_files(self, genre):
        return len(os.listdir(f"{DATA_FOLDER}{genre}/"))

    # Return the 
    def get_files(self, genre):
        sub_folder = f"{DATA_FOLDER}{genre}/"
        files = os.listdir(sub_folder)
        collection = [np.load(f"{sub_folder}{file}") for file in files]

        return np.array(collection)
