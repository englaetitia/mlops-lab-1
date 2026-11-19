# Lab 1 - git / dvc and data preparation

**Repository:** https://github.com/englaetitia/mlops-lab-1
**Author:** Laetitia Daou

## Solution adopted for the data size

I used **solution 2: reduce the data folder size.**

The full Food-11 dataset is too big to push easily to DagsHub. So I kept the full download in `data_local/food11_raw/`, which is in `.gitignore` and is not tracked by dvc.

Then I wrote a small script, `src/food11/subset_raw.py`, that copies only **5 images per category per split** into `data/food11_raw/`. It keeps the same folders (`training`, `evaluation`, `validation`) and the same file names (`<category>_<id>.jpg`). This smaller `data/` folder is what dvc tracks and pushes.

```
data_local/food11_raw/        full dataset, gitignored, not tracked by dvc
data/food11_raw/              5 images x 11 categories x 3 splits = 165 images, tracked by dvc
data/food11_processed/        made by src/food11/data.py
data/food11_processed_mini/   made by src/food11/data.py
```

The idea of the lab stays the same: git keeps the pointer file, dvc keeps the data, and I can still go back to an old version of `data/` using that pointer. The number of images doesn't change how it works.

---

## Question 1 - Observe the files created by `uv init`. What do they contain?

After `uv init` my repo had these files:

- `pyproject.toml`: the project description. It has the name, the version, the Python version needed (`requires-python`) and the list of `dependencies`. When I run `uv add pillow`, uv writes pillow in this file.
- `.python-version`: the Python version to use for this project (3.10 here), so everyone uses the same one.
- `README.md`: empty for now.
- `src/mlops_lab_1/__init__.py`: a small example package with a `main()` function that prints "Hello from mlops-lab-1!". My project uses the `src/` layout, which is why there is no `main.py` at the root. `pyproject.toml` also has a `[project.scripts]` entry pointing to this `main()` and a `[build-system]` section (`uv_build`), so the project can be installed as a package.

Two more things appear the first time you run `uv add`, `uv sync` or `uv run`:

- `uv.lock`: the exact version and hash of every package, including the dependencies of my dependencies. This is what makes the environment **reproducible**. It must be committed to git.
- `.venv/`: the virtual environment with the installed packages. It is big, it depends on the machine, and it can be rebuilt from `uv.lock`. It must **not** go to git.

So it is like git and dvc: we commit the description of the environment, not the environment itself.

## Question 2 - What files does `dvc init` create? What are they for? Which ones go to git?

After `dvc init` I got:

```
.dvc/
.dvc/config
.dvc/.gitignore
.dvc/tmp/
.dvcignore
```

- `.dvc/config`: the dvc settings for the project (remotes, default remote...). It is shared with the team, so it **goes to git**.
- `.dvc/.gitignore`: made by dvc. It tells git to ignore `/config.local`, `/tmp` and `/cache`. It **goes to git**.
- `.dvcignore`: like `.gitignore` but for dvc. It lists files dvc should skip. It **goes to git**.
- `.dvc/tmp/`: temporary files (locks, internal state). **Not in git.**
- Later, `.dvc/cache/` appears after the first `dvc add`. It holds the real data files on my machine. **Not in git.**
- `.dvc/config.local`: settings for this machine only, for example passwords. **Not in git.**

Right after `dvc init`, `git status` already showed `.dvc/config`, `.dvc/.gitignore` and `.dvcignore` as staged. So `git add .dvc .dvcignore` just confirms it. Thanks to `.dvc/.gitignore`, `git add .dvc` can't add the cache by mistake.

## Question 3 - Where are the credentials stored? What are the other options besides `--global`? Should they be pushed to GitHub?

With `--global`, dvc saves the settings in my user config file, outside the repo:

- Windows: `%LOCALAPPDATA%\iterative\dvc\config`
- Linux: `~/.config/dvc/config`
- macOS: `~/Library/Application Support/dvc/config`

The options are:

- `--project` (the default): writes to `.dvc/config`. Shared with everyone and **tracked by git**.
- `--local`: writes to `.dvc/config.local`. Only this repo on this machine, ignored by git.
- `--global`: the user config file above. All my repos, outside the repo.
- `--system`: a config for all users of the machine. Outside the repo.

**The credentials should never be pushed to GitHub.** Anything committed stays in the git history, even if you delete it in a later commit, and anyone with access to the repo can read it. Passwords and tokens go in `--local` or `--global`. Only the non-secret settings (remote name and URL) go in `.dvc/config`.

**Problem I had:** the lab says `dvc remote add --global origin ...`. That puts the remote URL in my user config too, so `.dvc/config` stays empty and `git add .dvc/config` commits nothing. Someone cloning the repo would not get the remote. So I added the remote at project level and only kept the secrets local:

```bash
dvc remote add -d origin https://dagshub.com/englaetitia/mlops-lab-1.dvc   # goes to .dvc/config (committed)
dvc remote modify origin --local auth basic                                # goes to .dvc/config.local
dvc remote modify origin --local user englaetitia
dvc remote modify origin --local password <dagshub-token>
```

My committed `.dvc/config` only has the remote name and URL, no password.

## Question 4 - Look at the `.gitignore` file. What happened?

After `dvc add data`, a new line was added at the end of `.gitignore`:

```
.venv/
__pycache__/
*.pyc
data_local/
/data          <- added by dvc
```

dvc added `/data` itself, so git ignores the data folder. Git and dvc should not both track the same files. If git also stored the images, the repo would get huge, which is exactly what dvc is there to avoid.

So now:
- **git** tracks `data.dvc`, a small text file
- **dvc** tracks what is inside `data/`

## Question 5 - Is there a `.dvc` file? What does it contain?

Yes, `data.dvc` at the root of the repo:

```yaml
outs:
- md5: 52f75517b85801c97600836f5c134a1f.dir
  size: 11338724
  nfiles: 165
  hash: md5
  path: data
```

- `path`: the folder it points to (`data`)
- `md5`: the hash of the whole folder. The `.dir` at the end means it is a folder, not a single file.
- `size` and `nfiles`: the total size (about 10.8 MB) and the number of files (165)
- `hash`: the hash algorithm used (md5)

This `.dir` hash is also the name of a file in the dvc cache (`.dvc/cache/files/md5/52/f75517b85801c97600836f5c134a1f.dir`). That file is a list of every image in the folder with its own md5:

```json
[{"md5": "0053fd179e1fedbcb52e05a00b8a9ddd", "relpath": "food11_raw/evaluation/0_0.jpg"},
 {"md5": "a3cc937f0fa90451e033ec6df317ce7a", "relpath": "food11_raw/evaluation/0_1.jpg"},
 ...]
```

So a 5-line file in git is enough to find the exact 165 images. dvc stores files by their content (the md5), not by their name. If two files are the same they are stored once. If I change one image, only that image gets a new hash, and so does the folder.

## Question 6 - On GitHub: is the code there? Is the data there? Is there a file pointing to the data? And on DagsHub?

**On GitHub:** the code is there (`src/food11/*.py`, `pyproject.toml`, `uv.lock`, `.dvc/config`, `.dvcignore`, `.gitignore`). **The data is not**: there is no `data/` folder because it is in `.gitignore`. The file that points to the data is **`data.dvc`**.

**On DagsHub:** the data is there, in the repo's Data / DVC section. The remote doesn't store a normal folder tree. It stores files named by their md5, like:

```
files/md5/75/a8cc6472c5c69bde883e466d379c5d
files/md5/87/dc163017b8644662e06e3a04947b1c
```

(the first 2 characters of the hash are the folder name). DagsHub reads the `.dir` list and shows the normal folders in its web UI, so you see the code and the data of the same commit in one place.

## Question 7 - Clone the repo in a new folder. Is the data folder there? Which dvc command gets it?

**No.** The new clone has the code and `data.dvc`, but no `data/` folder, because git only has the pointer.

To get the data:

```bash
dvc pull
```

`dvc pull` does two things: `dvc fetch` downloads the files listed in `data.dvc` from the remote into `.dvc/cache`, and `dvc checkout` puts them in `data/`.

If the files are already in the local cache, `dvc checkout` alone is enough. That is the case in Q8.

## Question 8 - After `git checkout <old-commit>` and `dvc checkout`, are `food11_processed` and `food11_processed_mini` still there?

**No, but only after `dvc checkout`.** Here is what I saw:

```
> git log --oneline -- data.dvc
aa6540e Add food11_processed and food11_processed_mini
7dd5c51 Track data folder with dvc

> git checkout 7dd5c51
HEAD is now at 7dd5c51 Track data folder with dvc

> dir data
food11_processed   food11_processed_mini   food11_raw      <- still there

> type data.dvc
  md5: 52f75517b85801c97600836f5c134a1f.dir
  nfiles: 165                                              <- old pointer

> dvc checkout
M       data\

> dir data
food11_raw                                                 <- now they are gone
```

After `git checkout` alone, the processed folders are **still on disk**. Git can't remove them because they are ignored, so git doesn't know about them. Git only changed `data.dvc` back to the old hash.

Then `dvc checkout` makes `data/` match that old pointer. It sees that the processed folders are not in the old version, so it removes them and only `food11_raw/` stays. Nothing is lost, the newer files are still in `.dvc/cache` and on DagsHub.

Going back to main brings them back:

```
> git checkout main
> dvc checkout
Applying changes |330 [00:01, 284file/s]
M       data\
> dir data
food11_processed   food11_processed_mini   food11_raw
```

330 files (165 processed + 165 mini) came back in about one second without internet, because they were still in the local cache. That's also why a fresh clone (Q7) needs `dvc pull` and not just `dvc checkout`.

**What I learned:** `git checkout` changes the code and the pointer, and `dvc checkout` changes the data to match the pointer. If you only do the first one, you run old code on new data without noticing.

---

## Final state at the end of lab 1

```yaml
outs:
- md5: 578b872da1d78eaf7658df6e47a0963c.dir
  size: 13638728
  nfiles: 495
  hash: md5
  path: data
```

495 files = 165 raw + 165 processed + 165 mini (about 13 MB). The git repo itself is still only text: code, `pyproject.toml`, `uv.lock`, `.dvc/config` and this small `data.dvc`.

(In lab 2 I rebuilt the processed datasets bigger so the model could actually train, so the current `data.dvc` has 9635 files. That is explained in `lab2.md`.)

## Files I made in this lab

- `src/food11/subset_raw.py`: makes the small `data/food11_raw` (5 images per category per split) from the full dataset in `data_local/food11_raw`. Needed because I chose solution 2.
- `src/food11/data.py`: the script asked in the lab. It resizes every image to 128x128 and puts each image in a folder named after its category (`<split>/<Category>/`), which is the layout `torchvision.datasets.ImageFolder` (and so ResNet training) needs. It creates `food11_processed` and `food11_processed_mini`.
- `lab/lab1.md`: this report.

**About `food11_processed_mini`:** `data.py` keeps at most 100 images per category in the mini dataset. Since my dvc raw data only has 5 per category, mini and processed were the same at this point. To see a difference you can run:

```bash
uv run python ./src/food11/data.py --mini-per-category 2
# or use the full dataset:
uv run python ./src/food11/data.py --source data_local/food11_raw
```
