import os
import argparse
import audiofile
import requests
import zipfile
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
from librosa.feature import melspectrogram
from librosa import to_mono

BASE_PATH = "data/"
SUB_FOLDER_PATH = "data/fma/"
METADATA_PATH = "data/fma/fma_metadata/tracks.csv"
GENRES_PATH = "data/fma/fma_metadata/genres.csv"
DATASET_LINK = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
METADATA_LINK = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"

parser = argparse.ArgumentParser(
    prog="1D DDPM Data Processing Script"
)

parser.add_argument("--n_fft", default=4096, required=False)
parser.add_argument("--hop_length", default=2585, required=False)
parser.add_argument("--n_mels", default=256, required=False)
parser.add_argument("--length", default=512, required=False)
parser.add_argument("--download", default=False, required=False)

args = parser.parse_args()

def threaded_fn(genre):
    files = os.listdir(f"{SUB_FOLDER_PATH}genres/{genre}/")

    if not os.path.exists("data/fma/spectrograms/"):
        os.mkdir("data/fma/spectrograms/")
    if not os.path.exists(f"data/fma/spectrograms/{genre}/"):
        os.mkdir(f"data/fma/spectrograms/{genre}/")

    for file in tqdm(files, total=len(files), desc=f"{genre}", leave=False):
        try:
            signal, sr = audiofile.read(f"{SUB_FOLDER_PATH}genres/{genre}/{file}")
            signal = to_mono(signal)
        except RuntimeError:
            print("\nInvalid MP3 File\n")
            continue
        if sr != 44100:
            continue
        spectrogram = melspectrogram(
            y=np.array(signal), 
            sr=sr, 
            n_fft=args.n_fft, 
            hop_length=args.hop_length, 
            n_mels=args.n_mels,
            dtype=np.float32
            ).squeeze()
        
        # Remove occasional extra dim:
        if spectrogram.shape[-1] == args.length:
            np.save(f'data/fma/spectrograms/{genre}/{file.split(".")[0]}.npy', spectrogram)

# Clean up genres column:
def clean_genres(x):
    # Remove genres without genre id:
    if x[0] != "[":
        return np.nan
    # Extract id:
    x = x.split(",")[0]
    x = x.replace("[", "")
    x = x.replace("]", "")

    return x

# Fix genre names:
def clean_names(x):
    x = str(x).replace(":", "-")
    x = x.replace("/", "-")

    return x

def main():
    # Prepare dataset:
    if args.download:
        # Create data folder:
        if not os.path.exists(BASE_PATH):
            os.mkdir(BASE_PATH)

        # Download dataset:
        response = requests.get(DATASET_LINK, stream=True)
        data_name = "data.zip"
        
        if not os.path.exists(data_name):
            with open(data_name, "wb") as handle:
                for data in tqdm(response.iter_content(chunk_size=1024*1024), desc="Downloading dataset", unit="MB"):
                    handle.write(data)
        
        if not os.path.exists(SUB_FOLDER_PATH):
            print("INFO: Unzipping dataset")
            with zipfile.ZipFile(data_name, "r") as zip_ref:
                zip_ref.extractall(BASE_PATH)

            os.rename(BASE_PATH + "fma_medium", BASE_PATH + "fma")

        # Download metadata:
        response = requests.get(METADATA_LINK, stream=True)
        metadata_name = "metadata.zip"

        if not os.path.exists(metadata_name):
            with open(metadata_name, "wb") as handle:
                for data in tqdm(response.iter_content(chunk_size=1024*1024), desc="Downloading metadata", unit="MB"):
                    handle.write(data)

        if not os.path.exists(METADATA_PATH):
            print("INFO: Unzipping metadata")
            with zipfile.ZipFile(metadata_name, "r") as zip_ref:
                zip_ref.extractall(BASE_PATH)
            
            os.rename(BASE_PATH + "fma_metadata", SUB_FOLDER_PATH + "fma_metadata")

        # Remove leftover files:
        if os.path.exists("data.zip"):
            os.remove("data.zip")
        if os.path.exists("metadata.zip"):
            os.remove("metadata.zip")
        if os.path.exists("data/fma_medium/"):
            shutil.rmtree("data/fma_medium/")

        print("INFO: Dataset downloaded successfully")

    metadata = pd.read_csv(METADATA_PATH, dtype=str)
    genres = pd.read_csv(GENRES_PATH, dtype=str)

    # Reassign column headers:
    metadata.columns = metadata.iloc[0]
    metadata.columns.values[0] = "track_id"
    metadata.drop(index=[0,1], inplace=True)

    metadata = metadata[["track_id", "title", "genres"]]

    metadata["genres"] = metadata["genres"].apply(clean_genres)

    # Join metadata and genres:
    metadata = metadata.merge(genres, how="left", left_on="genres", right_on="genre_id")
    metadata = metadata.rename(columns={"title_y": "genre"})

    metadata["genre"] = metadata["genre"].apply(clean_names)

    # Create folders for each genre:
    unique_genres = metadata["genre"].unique()
    with open("data/fma/available_genres.txt", "w") as file:
        for genre in unique_genres:
            file.write(str(genre) + "\n")
    
    if not os.path.exists("data/fma/included_genres.txt"):
        with open("data/fma/included_genres.txt", "w") as file:
            for genre in unique_genres:
                file.write(str(genre) + "\n")

    if not os.path.exists(SUB_FOLDER_PATH + "genres/"):
        os.mkdir(SUB_FOLDER_PATH + "genres/")

    for genre in unique_genres:
        if not os.path.exists(f'data/fma/genres/{genre}/'):
            os.mkdir(f'data/fma/genres/{genre}/')

    # Move files to correct folders:
    sub_folders = os.listdir(SUB_FOLDER_PATH)
    for folder in tqdm(sub_folders, total=len(sub_folders), desc="Folders moved"):
        try:
            files = os.listdir(SUB_FOLDER_PATH + folder + "/")
        except:
            print("Operation finished")
            break

        if len(files) == 0:
            print("No files found in original location")
            break

        for file in files:
            track_id = int(file.split(".")[0])
            row = metadata[metadata["track_id"] == str(track_id)]
            genre = str(row["genre"].values[0])
            os.rename(SUB_FOLDER_PATH + folder + "/" + file, SUB_FOLDER_PATH + "genres/" + genre + "/" + file)

    # Convert audio to spectrograms:
    print("INFO: Generating spectrograms")

    pool = ThreadPool(cpu_count() - 2)
    pool.map(threaded_fn, unique_genres)

    print("INFO: Processing complete")

if __name__ == "__main__":
    main()