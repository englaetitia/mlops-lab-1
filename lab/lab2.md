# Lab 2 - Model training and experiment tracking with MLflow

**Repository:** https://github.com/englaetitia/mlops-lab-1
**Author:** Laetitia Daou

## Note on the dataset

In lab 1 I used the small-data option, so `food11_processed_mini` only had 5 images per category. That is too few to compare learning rates. So before training I rebuilt the processed datasets from the full local copy:

```bash
uv run python ./src/food11/data.py --source data_local/food11_raw \
    --per-category 200 --mini-per-category 100
```

Now `food11_processed_mini` has 100 images per category per split (1,100 training images) and `food11_processed` has 200 per category (2,200 training images). I added the `--per-category` option to `data.py` for this, so the datasets stay small enough to push to DagsHub but big enough to train on. `data/food11_raw` did not change.

---

## Question 1 - Look at `pyproject.toml` and `uv.lock`. What changed?

**In `pyproject.toml`**, the new packages were added to `dependencies`:

```toml
dependencies = [
    "mlflow>=3.16.0",
    "pillow>=12.3.0",
    "scikit-learn>=1.7.2",
    "torch>=2.0",
    "torchvision>=0.15",
]
```

I first added the CPU-only PyTorch index like the lab says. Then I switched to the CUDA build, because my laptop has an NVIDIA GPU (RTX 3050) and training is much faster on it. So the file got a `[[tool.uv.index]]` block and a `[tool.uv.sources]` block that tell uv to download torch and torchvision from the PyTorch CUDA index instead of PyPI:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
```

`explicit = true` means this index is only used for the packages listed in `[tool.uv.sources]`. Everything else still comes from PyPI.

(In lab 3 I changed this again so Windows gets CUDA and Linux gets CPU. That is explained in `lab3.md`.)

**`uv.lock` became much bigger**: it went from 2 packages to 133. My 5 dependencies pull in many other packages, like `numpy`, `pandas`, `sqlalchemy`, `alembic`, `flask`, `pyarrow`, `scipy`, `sympy`, `networkx`...

The difference between the two files:
- `pyproject.toml` = what **I asked for** (a few names, with loose versions like `>=2.0`)
- `uv.lock` = what uv **actually installed** (every package with its exact version, hash and source)

The lock file is what makes the environment reproducible, so it goes in git. `.venv/` doesn't.

## Question 2 - What are `--backend-store-uri` and `--default-artifact-root` for? What is the difference between metadata and artifacts?

- `--backend-store-uri sqlite:///mlflow.db` is where MLflow keeps the **metadata**: experiments, runs, params, metrics, tags, status, start and end time. It is a database, here a SQLite file.
- `--default-artifact-root ./mlruns` is where MLflow keeps the **artifacts**: the files a run produces, like the saved model, plots, `requirements.txt`...

I opened `mlflow.db` after a few runs. It has tables like `experiments`, `runs`, `params`, `metrics`, `tags`, `model_versions`. Each metric value is one row `(key, value, step, timestamp, run_uuid)`. There are no files inside it.

**The difference:** metadata is small and structured, so you can search and sort it. That's what lets the UI sort runs by `val_accuracy` or compare runs. Artifacts are big files that a database can't search. My `model.pth` alone is about 45 MB. So the database only keeps the path to the artifact, and the file itself stays on disk.

It's the same idea as git and dvc in lab 1: small info in one place, big files somewhere else.

## Question 3 - Why shouldn't `mlflow.db` and `mlruns/` be tracked by git? And why not by dvc either?

**Not git:**
1. `mlflow.db` is a binary file. Git saves a full new copy every time it changes and `git diff` shows nothing useful.
2. `mlruns/` has the saved models (about 45 MB per run). Big files like that are exactly what lab 1 said should not go in git.
3. Both change on every training run, even when the code doesn't change. The repo would never be clean, and two people training at the same time would get a conflict on a binary file that can't be merged.

**Not dvc either:** dvc is for data that goes with a specific commit, like a dataset the code uses. When you check out a commit, you get the data that matches it. MLflow history is different:
- It's a log of everything that happened. It only grows: running the code again adds new runs, it doesn't recreate the same file.
- It's not linked to one commit. One `mlflow.db` has runs from many commits, so saving it under one commit doesn't make sense.
- MLflow already manages this data with its own storage and UI. Tracking it with dvc too would mean two tools managing the same thing.

In a real team the tracking server is shared and remote (DagsHub has a hosted MLflow for example), so everyone logs to the same place. My local `mlflow.db` just replaces that server for the lab.

## Question 4 - What happens the first time you call `set_experiment` with a name that doesn't exist?

MLflow **creates the experiment** and doesn't give an error. All the next runs go into it.

I checked in the database:

```
experiments: [(0, 'Default'), (1, 'food11')]
```

`Default` (id 0) is created when the server starts. `food11` got id 1 when my training script first called `set_experiment("food11")`. It showed up right away in the left menu of the UI, with no runs yet.

When I call `set_experiment("food11")` again, MLflow just finds the existing experiment and doesn't make a second one.

## Question 5 - What is the difference between `log_param` and `log_metric`? Why does only `log_metric` take a `step`?

- A **param** is a value chosen **before** training that stays the same for the whole run, like `lr`, `batch_size`, `epochs`, `model`, `seed`. It is saved as text and logged once. Logging the same param again with a different value gives an error.
- A **metric** is a number **produced during** training, like `train_loss`, `val_loss`, `val_accuracy`. It can be logged many times, and each value is saved with a `step` and a timestamp.

**Why `step` only for metrics:** a param doesn't change during the run. `lr` is 0.001 from start to finish, so "lr at epoch 3" means nothing. A metric changes at every epoch, and what's interesting is the curve. `step` is the x-axis of that curve (here, the epoch number). With it the UI can draw loss vs epoch, and you can see if the model is still learning or starting to overfit.

Simple rule: if the value can change while the run is going, it's a metric; if not, it's a param. That's why I log `train_images` and `device` as params (known before training) and `training_seconds` as a metric (only known at the end).

## Question 6 - Find the params, metric charts and model artifact in the UI. Where does the model actually live on disk?

**In the UI**, when I open a run:
- **Parameters:** the 12 values I log with `mlflow.log_params({...})` (dataset, epochs, lr, batch_size, optimizer, model, pretrained, freeze_backbone, num_classes, train_images, seed, device)
- **Metrics:** each metric with its last value. Clicking on one shows a chart of the value per step (per epoch). `test_accuracy` is a single point because I log it once at the end.
- **Artifacts:** a `model` folder with `MLmodel`, `conda.yaml`, `python_env.yaml`, `requirements.txt`, `input_example.json` and `data/model.pth`

**On disk**, the model is under the folder given in `--default-artifact-root`:

```
mlruns/<experiment_id>/models/m-<model_id>/artifacts/
├── MLmodel
├── data/model.pth          <- the actual weights (~45 MB)
├── data/pickle_module_info.txt
├── conda.yaml
├── python_env.yaml
├── requirements.txt
├── input_example.json
└── serving_input_example.json
```

For my `food11` experiment (id 1) that's `mlruns/1/models/m-<id>/artifacts/data/model.pth`.

The database only has the path, the file is on disk (like I said in Q2). The `MLmodel` file is what makes this more than just weights: it says which flavors the model has (`pytorch` and `python_function`), the torch version, the input/output signature and the `run_id` it came from. That's what lets me load and serve the model in the next lab without rewriting the training code.

(This folder layout is MLflow 3. In MLflow 2 the model was in `mlruns/<experiment_id>/<run_id>/artifacts/model/`.)

## Question 7 - Which learning rate gave the best `val_accuracy`? Is higher always better?

All runs: 5 epochs, pretrained resnet18, Adam, `food11_processed_mini` (1,100 training images), on my RTX 3050. I ran `lr=0.001, batch_size=32` twice with the same settings.

| lr | batch_size | val_accuracy | test_accuracy | val_loss | time (s) |
|---|---|---|---|---|---|
| 0.0001 | 32 | **0.7482** | **0.7801** | **0.7931** | 32.5 |
| 0.001 | 64 | 0.6150 | 0.6469 | 1.4524 | 30.7 |
| 0.001 | 32 (run 1) | 0.5593 | 0.5776 | 1.5400 | 29.8 |
| 0.001 | 32 (run 2, same settings) | 0.4891 | 0.5182 | 2.0157 | 31.5 |
| 0.01 | 32 | 0.2336 | 0.2381 | 2.3300 | 33.1 |

**The best was the smallest learning rate, 0.0001**, with 74.8% validation accuracy, about 3 times better than 0.01 (23.4%).

**So no, higher is not better.** The bigger the learning rate, the lower the accuracy and the higher the `val_loss`. At 0.01 the model only gets 23.4%. That's better than random guessing (1/11 = 9.1%), but each update is so big that it overshoots and undoes the previous one.

The reason is that the model is **pretrained**. Its layers already learned useful features on ImageNet. Only the new last layer (11 classes) really needs to learn. With a big learning rate, the big updates also hit the pretrained layers and destroy what they learned. Small steps keep those features, which is the point of fine-tuning.

But 0.0001 is only the best of the values I tried, not necessarily the best possible. Its `val_loss` was still going down at epoch 5, so it was probably still improving. Trying 0.00001, or training 0.0001 for more epochs, would be the next thing to test.

## Question 8 - What pattern do you see in the parallel coordinates plot (`lr`, `batch_size`, `val_accuracy`)?

The lines are sorted almost completely by `lr`. The 0.0001 line goes to the top of `val_accuracy`, the 0.001 lines are in the middle, and the 0.01 line goes to the bottom. `batch_size` doesn't sort the lines: the 64 line ends up in the middle of the 0.001 group.

At first sight that says "lr matters, batch size doesn't". But the two identical runs change how I read it. With the **same settings and the same seed** (`lr=0.001, batch_size=32`), I got **0.4891 and 0.5593**, a 7-point difference just from randomness. On GPU, cuDNN picks some algorithms that are not exactly reproducible, so `torch.manual_seed` doesn't guarantee the same result. So about 7 points is the noise in my experiment.

Compared to that noise:
- the `lr` effect (from 23% to 75%, about 51 points) is much bigger, so it's **real**
- the `batch_size` effect (61.5% with 64 vs 49-56% with 32, so 6 to 13 points) is about the same size as the noise, so I **can't conclude** anything about it from these runs

To really know, I'd need to repeat each setting several times and compare the averages. Also, lr and batch size are linked (a bigger batch gives a less noisy gradient and can often handle a slightly bigger lr), and changing one at a time can't show that.

## Question 9 - Best run by `val_accuracy`

Sorted by `val_accuracy` descending, the best run is:

- **Run ID:** `36d97adc0b7a4126b445dc10ea66b83d`
- **Params:** `lr=0.0001`, `batch_size=32`, `epochs=5`, pretrained resnet18, Adam
- **val_accuracy:** 0.7482
- **test_accuracy:** 0.7801
- **val_loss:** 0.7931

I'll use this run ID in lab 3 to register the model.

It is the best on all three numbers (highest val and test accuracy, lowest val loss), so the choice doesn't depend on which metric I sort by.

Two remarks:
- `val_accuracy` is the number I used to **choose** the run, so it's a bit optimistic. The `test_accuracy` on the `evaluation` split wasn't used for choosing, so that's the number to report.
- Here `test_accuracy` is about 3 points **higher** than `val_accuracy` in every run. That's not because of the model: the mini dataset takes the first 100 files of each category (sorted by name), not a random sample, so the validation and evaluation splits are a bit different and evaluation is slightly easier. It doesn't change the ranking because it affects every run the same way. But 78% means "on this subset", not on the full Food-11.

---

## Files I made in this lab

- `src/food11/train.py`: fine-tunes a pretrained resnet18 on Food-11. It logs the params, the metrics of each epoch, the final test accuracy and the model to MLflow.
- `src/food11/data.py`: updated with `--per-category` to limit the size of the processed dataset separately from the mini one.
- `lab/lab2.md`: this report.

### Notes on the code

- **Why only the last layer is replaced:** resnet18 ends with a `fc` layer of 512 to 1000 (ImageNet classes). I replace it with `nn.Linear(512, 11)`. All the other layers are kept, so the model already knows edges, textures and shapes, and only learns how to map them to the 11 food categories. With `--freeze-backbone`, only this last layer is trained, which is faster on CPU but a bit less accurate.
- **Why ImageNet mean and std for normalization:** the pretrained weights were trained on images normalized with these values. If the input is scaled differently, the features work much worse.
- **Why `log_model` gets an `input_example`:** in MLflow 3 the default save format (`pt2`) needs an example input to run the model once, otherwise it refuses to save. I pass one real validation image and use the `pickle` format. The example also saves the model's signature (input/output shape), which I need to serve the model in lab 3.
