import torch
import os
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm

DATA_FOLDER = "data/fma/spectrograms/"

# NEED TO UPDATE
EXAMPLE_FOLDER = "data/fma/spectrograms/Rock/"
GENRES_PATH = "data/fma/included_genres.txt"
SAMPLE_RATE = 44100

class FMA(torch.utils.data.Dataset):
    def __init__(self, workers=10, normalize=True):
        files = os.listdir(EXAMPLE_FOLDER)
        example = np.load(f"{EXAMPLE_FOLDER}{files[0]}")
        sample_shape = example.shape

        print(f"INFO: Data sample shape: {sample_shape}")

        pool = ThreadPool(workers)
        self.normalize = normalize
        self.data = []
        self.sample_rate = SAMPLE_RATE
        self.mmap_samples = None
        self.mmap_path = "data/fma/memmap.dat"
        self.size = 0

        # Collect specified genres:
        with open(GENRES_PATH, "r") as file:
            lines = file.readlines()
            genres = [line.strip() for line in lines]
        
        # Remove empty folders:
        empty_folders = []
        for genre in genres:
            files = os.listdir(f"{DATA_FOLDER}{genre}/")

            if len(files) == 0:
                empty_folders.append(genre)
        
        genres = [genre for genre in genres if genre not in empty_folders]
        
        if len(empty_folders) != 0:
            os.remove(GENRES_PATH)

            with open(GENRES_PATH, "w") as file:
                for genre in genres:
                    file.write(str(genre) + "\n")

        # Determine size of dataset:
        print("INFO: Counting files")
        for result in pool.map(self.count_files, genres):
            self.size += result

        print(f"INFO: Counted {self.size} files")
        
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

        print(f"INFO: Length of memmap: {len(self.mmap_samples)}")

        # Collect songs in each genre:
        if mode in "w+":
            num_empty = 0

            print("INFO: Constructing memmap")

            index = 0
            for result in tqdm(pool.map(self.get_files, genres), total=len(genres), desc="Genre"):
                length = len(result)
                
                if length == 0:
                    continue

                for sample in result:
                    if len(sample) == len(sample[sample == 0]):
                        num_empty += 1

                self.mmap_samples[index:index + length][:] = result[:]

                index += length

            print(f"Empty samples: {num_empty}")
            

    def __getitem__(self, idx):
        sample = self.mmap_samples[idx]

        if self.normalize:
            min = sample.min((1), keepdims=True)
            max = sample.max((1), keepdims=True)
            
            sample = (sample - min) / (max - min)
            sample = np.nan_to_num(sample)

        return sample
    
    def __len__(self):
        return len(self.mmap_samples)

    # Return the number of files per genre:
    def count_files(self, genre):
        files = os.listdir(f"{DATA_FOLDER}{genre}/")

        count = 0
        for file in files:
            sample = np.load(f"{DATA_FOLDER}{genre}/{file}")
            if len(sample) != len(sample[sample == 0]):
                count += 1

        return count

    # Return the files in the folder for the specified genre:
    def get_files(self, genre):
        sub_folder = f"{DATA_FOLDER}{genre}/"
        files = os.listdir(sub_folder)

        collection = []
        for file in files:
            sample = np.load(f"{sub_folder}{file}")

            if len(sample) != len(sample[sample == 0]):
                collection.append(sample)

        return np.array(collection)
