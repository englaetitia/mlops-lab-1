# Lab 2 — Model training and experiment tracking with MLflow

**Repository:** https://github.com/englaetitia/mlops-lab-1
**Author:** Laetitia Daou

## Note on the dataset used

Lab 1 used the reduced-data option, so `data/food11_processed_mini` held only 5 images
per category — far too few for a learning-rate comparison to mean anything. Before
training, the mini dataset was rebuilt from the full local copy:

```bash
uv run python ./src/food11/data.py --source data_local/food11_raw \
    --per-category 200 --mini-per-category 100
```

`data/food11_processed_mini` now holds 100 images per category per split
(1,100 training images), and `data/food11_processed` holds 200 per category
(2,200 training images). `data/food11_raw` is unchanged from Lab 1. The `--per-category`
flag was added to `data.py` for this, so the processed datasets stay small enough to push
to DagsHub while still being large enough to train on.

---

## Question 1 — Look at `pyproject.toml` and `uv.lock`. What changed?

**`pyproject.toml`** gained four entries under `dependencies`:

```toml
dependencies = [
    "pillow>=...",
    "mlflow>=...",
    "torch>=...",
    "torchvision>=...",
    "scikit-learn>=...",
]
```

plus the two blocks that redirect torch to the CPU-only wheel index:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }
```

`explicit = true` is what makes this work: the index is used *only* for packages that
name it in `[tool.uv.sources]`, so everything else still resolves from PyPI.

**`uv.lock`** grew enormously — from a handful of packages to several hundred. Four
direct dependencies drag in their whole transitive closure: `numpy`, `pandas`,
`sqlalchemy`, `alembic`, `flask`, `gunicorn`/`waitress`, `pyarrow`, `docker`, `sympy`,
`networkx`, `filelock`, `scipy`, `joblib` and so on.

The distinction is the important part:

- `pyproject.toml` records **what I asked for** — four names, loose version constraints.
- `uv.lock` records **what was actually resolved** — every package, its exact version,
  its hash, and which index it came from.

Only the lock file makes the environment reproducible, which is why it is committed to
git while `.venv/` is not.

## Question 2 — What are `--backend-store-uri` and `--default-artifact-root` for? What is the difference between metadata and artifacts?

| Flag | Stores | In this lab |
|---|---|---|
| `--backend-store-uri` | **Metadata**: experiments, runs, params, metrics, tags, run status and timestamps. Structured records in a database. | `sqlite:///mlflow.db` |
| `--default-artifact-root` | **Artifacts**: arbitrary files produced by a run — serialised models, plots, environment files, sample inputs. | `./mlruns` |

Inspecting `mlflow.db` after a few runs shows exactly this split — tables named
`experiments`, `runs`, `params`, `metrics`, `latest_metrics`, `tags`, `model_versions`.
Every metric point is a row: `(key, value, step, timestamp, run_uuid)`. Nothing binary
lives there.

**The difference that matters:**

Metadata is small, structured and **queryable** — it is what makes the runs table
sortable, the compare page possible, and "show me every run with `lr` < 0.01 ordered by
`val_accuracy`" a single SQL query. A metric is a handful of numbers.

Artifacts are large, opaque **blobs** the database has no way to index. My `model.pth`
alone is ~45 MB. Putting that in SQLite would make every query slow and the file
unmanageable, so the database stores only a *path* and the bytes live on the filesystem.

It is the same separation as Lab 1's git/dvc split, one layer up: small structured facts
in a database that can be searched, big binary objects in content storage that is only
ever fetched by reference.

## Question 3 — Why shouldn't `mlflow.db` and `mlruns/` be tracked by git, and why not by dvc either?

**Not git**, for three reasons:

1. `mlflow.db` is a **binary** SQLite file. Git stores a whole new copy on every change
   and `git diff` shows nothing useful. After twenty runs the repo carries twenty
   near-identical multi-megabyte blobs.
2. `mlruns/` contains **model weights** — ~45 MB per run for resnet18. That is exactly
   the kind of thing Lab 1 established should never enter git.
3. Both change on **every single run**, independently of any code change. The working
   tree would never be clean, and two people training simultaneously would produce a
   binary merge conflict that cannot be resolved.

**Not dvc either**, which is the subtler half of the question. dvc versions *data as an
input or output of a reproducible pipeline*: a dataset that belongs to a specific commit,
such that checking out that commit gives you exactly the files the code ran against.
MLflow run history is not that kind of thing:

- It is an **append-only log of what happened**, not an input. Re-running the code does
  not recreate it — it adds to it.
- It is not tied to one commit. A single `mlflow.db` accumulates runs from many commits,
  so pinning it to any one commit hash is meaningless.
- MLflow is already the system of record for this data, with its own storage backend and
  its own UI. Version-controlling it with dvc would mean two systems claiming ownership
  of the same state.

In a real setup the tracking server is remote and shared — DagsHub offers a hosted
MLflow endpoint — so the team writes to one central store rather than each committing
their own local copy. The local `mlflow.db` here is just a stand-in for that server.

## Question 4 — What happens the first time you call `set_experiment` with a name that doesn't exist?

MLflow **creates it** and returns the `Experiment` object; it does not error. Every
subsequent `mlflow.start_run()` in the process is then attached to it.

Confirmed in the backing database:

```
experiments: [(0, 'Default'), (1, 'food11')]
```

`Default` is experiment 0, created when the server first starts. `food11` was assigned
id 1 at the moment the first training run called `set_experiment("food11")`. In the UI it
appears in the left sidebar immediately, initially with no runs.

Calling `set_experiment("food11")` again later is a no-op lookup — it finds the existing
experiment rather than creating a second one, which is why the line can sit unguarded at
the top of the script.

## Question 5 — What is the difference between `log_param` and `log_metric`? Why does only `log_metric` take a `step`?

| | `log_param` | `log_metric` |
|---|---|---|
| What it is | A value **set before** training, fixed for the whole run | A value **produced during** training |
| Type | Stored as a string | Must be a float |
| How many times | Once per key — immutable; logging the same key twice is an error | Many times per key |
| Extra columns | none | `step` and `timestamp` |
| Examples | `lr`, `batch_size`, `epochs`, `model`, `seed` | `train_loss`, `val_loss`, `val_accuracy` |

**Why `step` exists only for metrics:** a parameter has no history — `lr` was 0.001 for
the entire run, so "lr at epoch 3" is not a meaningful question. A metric *is* a history:
`val_loss` has a different value at every epoch, and the interesting thing about it is
the shape of the curve. `step` is the x-axis. Without it MLflow would only be able to
show the last value; with it, the UI can draw loss against epoch, which is what tells you
whether the model is still learning or has started overfitting.

The practical rule follows: **if the value can change while the run is executing, it is a
metric; if it cannot, it is a param.** That is why my script logs `train_images` and
`device` as params (fixed before the first batch) but `training_seconds` as a metric
(only known at the end).

## Question 6 — Find the params, the metric charts, and the model artifact in the UI. Where does the model artifact actually live on disk?

**In the UI**, inside a run:

- **Parameters** table — the eleven values logged by `mlflow.log_params({...})`.
- **Metrics** tab — each metric listed with its latest value; clicking one opens a chart
  of value against `step`, i.e. against epoch. `test_accuracy` shows as a single point
  because it was logged once with no step.
- **Artifacts** tab — a `model` entry containing `MLmodel`, `conda.yaml`,
  `python_env.yaml`, `requirements.txt`, `input_example.json` and `data/model.pth`.

**On disk**, under the `--default-artifact-root` given to the server:

```
mlruns/<experiment_id>/models/m-<model_id>/artifacts/
├── MLmodel                      the manifest: flavors, signature, run_id, size
├── data/model.pth               the actual serialised weights (~45 MB)
├── data/pickle_module_info.txt
├── conda.yaml
├── python_env.yaml
├── requirements.txt
├── input_example.json
└── serving_input_example.json
```

So for the `food11` experiment (id 1) the model sits at
`mlruns/1/models/m-<hash>/artifacts/data/model.pth`.

Two things worth noticing. First, the database holds only the *path* — the bytes are on
the filesystem, exactly as described in Q2. Second, `MLmodel` is what makes this more
than a weights file: it records the flavors (`pytorch` and `python_function`), the
pytorch version, the model signature and the originating `run_id`, which is what lets the
next lab load and serve the model without knowing how it was built.

> Note: this layout is MLflow 3.x. Under MLflow 2.x the same files live at
> `mlruns/<experiment_id>/<run_id>/artifacts/model/` — keyed by run rather than by
> logged-model id.

## Question 7 — Which learning rate gave the best `val_accuracy`? Is higher always better?

All runs: 5 epochs, resnet18 pretrained, Adam, `food11_processed_mini`
(1,100 training images), on an RTX 3050.

| lr | batch_size | `val_accuracy` | `test_accuracy` | `val_loss` | seconds |
|---|---|---|---|---|---|
| 0.0001 | 32 | **0.7482** | **0.7801** | **0.7931** | 32.5 |
| 0.001 | 64 | 0.6150 | 0.6469 | 1.4524 | 30.7 |
| 0.001 | 32 | 0.5593 | 0.5776 | 1.5400 | 29.8 |
| 0.001 | 32 | 0.4891 | 0.5182 | 2.0157 | 31.5 |
| 0.01 | 32 | 0.2336 | 0.2381 | 2.3300 | 33.1 |

**The best learning rate was the lowest one, 1e-4**, at 74.8% validation accuracy — more
than three times the 23.4% that 1e-2 managed.

**No, higher is not better.** Accuracy falls monotonically as the learning rate rises,
and `val_loss` rises with it, which rules out the optimistic reading that the model is
merely slower to converge. At 1e-2 the model reaches 23.4% against a chance baseline of
1/11 ≈ 9.1% — it has learned something, but each update is large enough to overshoot and
undo most of the previous one.

The reason this band is lower than usual is that the network is **pretrained**. Its
convolutional filters already encode useful features; the only thing that genuinely needs
learning is the new 11-way head. A large learning rate applies those same large updates to
the backbone and destroys what ImageNet training produced — so the model has to relearn
from a worse starting point than it began with. Smaller steps preserve the transferred
features, which is exactly what fine-tuning is for.

One caveat, and it is the reason the two identical runs above matter: **1e-4 is the best
value tested, not necessarily the optimum.** `val_loss` was still falling at epoch 5 and
is by far the lowest of the five runs, so this configuration was probably still improving
when training stopped. Testing 1e-5, or running 1e-4 for more epochs, would be the honest
next step before declaring a winner.

## Question 8 — What pattern do the parallel coordinates show for `lr`, `batch_size` and `val_accuracy`?

The plot separates almost entirely on the `lr` axis. Lines entering at 0.0001 rise to the
top of the `val_accuracy` axis, lines at 0.001 land in the middle, and the single line at
0.01 drops to the bottom — a clean monotonic fan, with no crossing between learning-rate
groups. `batch_size` does the opposite: its two values do not sort the lines at all, and
the 64 line threads back into the middle of the 0.001 group rather than separating from
it.

Read naively that says "lr matters, batch size doesn't". The honest reading is more
careful, and the duplicated configuration is what makes the point:

**Two runs with identical parameters** (`lr=0.001, batch_size=32`, same seed) produced
**0.4891 and 0.5593** — a 7-point spread from nothing but nondeterminism. cuDNN selects
convolution algorithms at runtime and several are not bit-reproducible, so `torch.manual_seed`
does not pin the result on GPU. That 7 points is the **noise floor** of this experiment.

Measured against it:

- The `lr` effect (0.234 → 0.748, 51 points) is far larger than noise. Real.
- The `batch_size` effect (0.615 at 64 versus 0.489–0.559 at 32, so 6–13 points) sits
  **inside or barely above** the noise floor. Not established by this data.

So the parallel coordinates plot is telling the truth about `lr` and overstating its case
about `batch_size` — a single run per configuration cannot distinguish a small effect
from run-to-run variance. Settling it would take several repeats per configuration and a
comparison of the means. It is also worth remembering that the two are not independent: a
larger batch gives a less noisy gradient estimate, which usually tolerates a slightly
larger learning rate, and changing one variable at a time cannot reveal that interaction
at all.

## Question 9 — Best run by `val_accuracy`

Sorted by `val_accuracy` descending, the top run is:

- **Run ID:** `36d97adc0b7a4126b445dc10ea66b83d`
- **Params:** `lr=0.0001`, `batch_size=32`, `epochs=5`, resnet18 pretrained, Adam
- **`val_accuracy`:** 0.7482
- **`test_accuracy`:** 0.7801
- **`val_loss`:** 0.7931

Keep this run ID — Lab 3 registers this model.

It wins on every measure at once — highest validation accuracy, highest test accuracy and
lowest validation loss — which is reassuring, because a run that topped one metric while
losing on the others would suggest the ranking was an artefact of the metric chosen.

Two honest caveats to state alongside the number:

**`val_accuracy` is not an unbiased estimate of quality.** It is the quantity used to
*choose* this run, so it is optimistic by construction. `test_accuracy` is computed on the
`evaluation` split, which played no part in the selection, and is the number to report.

**Here `test_accuracy` is consistently *higher* than `val_accuracy`** — by about three
points, in every run. That is the opposite of the usual selection-bias direction and has
nothing to do with the model: the mini dataset takes the first 100 files per category by
filename rather than a random sample, so the validation and evaluation splits are not
statistically equivalent. The evaluation split happens to be slightly easier. It does not
change the ranking, since the effect is the same for every run, but it means 78% should be
read as "on this particular subset", not as an estimate of Food-11 performance.

---

## Files produced in this lab

| File | Purpose |
|---|---|
| `src/food11/train.py` | Fine-tunes a pretrained resnet18 on Food-11, logs params, per-epoch metrics, final test accuracy and the model to MLflow. |
| `src/food11/data.py` | Updated with `--per-category` so the processed datasets can be capped independently of the mini one. |
| `lab/lab2.md` | This file. |

### Implementation notes

**Why the final layer is replaced rather than the whole network retrained.** `resnet18`
ships with a `fc` layer of 512 → 1000 ImageNet classes. Swapping in
`nn.Linear(512, 11)` keeps every convolutional filter — edges, textures, shapes — and
learns only the mapping from those features to the eleven food categories. `--freeze-backbone`
takes this further and trains *only* that layer, which is much faster on CPU and slightly
less accurate.

**Why images are normalised with the ImageNet mean and standard deviation.** The
pretrained weights were learned on inputs scaled that way; feeding differently-scaled
inputs makes the features much less useful.

**Why `mlflow.pytorch.log_model` needs an `input_example` here.** MLflow 3's default
`pt2` serialization format traces the graph by running `forward` on a sample input, and
refuses to log without one. The script passes one real validation image and requests the
`pickle` format, which also records the model signature — needed to serve the model in
the next lab.
