# %%

%load_ext autotime

# %%
from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)
    
    
# %%
import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(),"rb") as f:
    ((x_train, y_train), (x_valid, y_valid,_)) = pickle.load(f, encoding="latin-1")
# %%
dir(as_posi)
# %%
print(pickle.load(f))


# %%
