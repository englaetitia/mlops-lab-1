# Lab 3 - Containerizing the model with Docker

**Repository:** https://github.com/englaetitia/mlops-lab-1
**Author:** Laetitia Daou

## Problems I had to fix before it worked

### 1. The tracking server had to serve the artifacts

In lab 2 I started the server with `--default-artifact-root ./mlruns`. With that option, MLflow saves the artifact location of each experiment as a **path on my laptop**:

```
(1, 'food11', 'C:\...\mlops-lab-1\mlruns\1')
```

When a client loads `models:/food11@champion`, it tries to open that path directly. That works on my laptop, but not inside a container, because the container doesn't have that folder. So the model can't be loaded, whatever the Dockerfile does.

The fix is to let the server send the artifacts over HTTP:

```bash
mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --artifacts-destination ./mlartifacts --serve-artifacts
```

New experiments then save a location like this instead:

```
(2, 'food11', 'mlflow-artifacts:/2')
```

and any client asks the server for the files. Old experiments keep their old path, so I renamed the lab 2 experiment to `food11-lab2` and trained the best settings again in a new `food11` experiment.

I also changed `--host 127.0.0.1` to `--host 0.0.0.0`. With `127.0.0.1` the server only accepts connections coming from the laptop itself, so the container would be refused.

### 2. The server had to accept the container's `Host` header

After that, the container still failed:

```
MlflowException: API request to endpoint /api/2.0/mlflow/registered-models/alias
failed with error code 403 != 200.
Response body: 'Invalid Host header - possible DNS rebinding attack detected'
```

MLflow checks the `Host:` header of every request to protect against DNS rebinding attacks. By default it only accepts localhost, 127.0.0.1 and private IP ranges. The container sends `Host: host.docker.internal:5000`, which is not in that list, so the request is refused. The connection itself worked: it was a clean 403, not a timeout.

```bash
mlflow server ... --allowed-hosts "localhost,localhost:*,127.0.0.1,127.0.0.1:*,host.docker.internal,host.docker.internal:*"
```

`--allowed-hosts` **replaces** the default list, it doesn't add to it. So I had to write localhost again, otherwise the UI in my browser would also get 403. I checked: with this list, `Host: host.docker.internal:5000` gets 200, and a host that is not in the list still gets 403.

### 3. CPU-only PyTorch in the container

In lab 2, `uv.lock` used the CUDA build of torch (about 3 GB of NVIDIA libraries) because my laptop has an RTX 3050. The container has no GPU, so that would be wasted space. `[tool.uv.sources]` accepts markers, so one lock file can work for both:

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cu128", marker = "sys_platform == 'win32'" },
    { index = "pytorch-cpu",   marker = "sys_platform != 'win32'" },
]
torchvision = [
    { index = "pytorch-cu128", marker = "sys_platform == 'win32'" },
    { index = "pytorch-cpu",   marker = "sys_platform != 'win32'" },
]
```

On Windows I train with CUDA, and the Linux image gets the CPU version. Same lock file, same `uv sync --frozen`.

---

## Question 1 - What version number did the model get? What is the difference between a run's logged model and a registered model?

**Version 1.**

- A **logged model** belongs to one run. You reach it with `runs:/<run_id>/model`. Its name and place depend on whichever run made it.
- A **registered model** has a fixed name (`food11`) and its own list of versions (1, 2, 3...). It doesn't depend on the experiment or the run. On top of the files, it can have versions, aliases, tags and a description. You reach it with `models:/food11@champion`.

**Registering doesn't copy anything.** The source of version 1 is `models:/m-cdc819e8c15e4a0cbc019d764ce11844`, which points to the same logged model. The registry is just a naming layer on top of artifacts that already exist.

In practice: a run ID changes every time I retrain, and I'd have to tell the serving code which run is the good one. `models:/food11@champion` never changes. The registry separates the experiments (lots of runs, most of them thrown away) from production (the one model we actually serve).

## Question 2 - What replaced the old stages? Why version separately from the run? Why are aliases more flexible than stages?

**Aliases** replaced the stages. They are names that you choose. The usual ones are `champion` (the model in production) and `challenger` (a model we're testing against it), but any name works: `canary`, `eu-prod`, `rollback`...

**Why version separately from the run:** a run tells you what happened during an experiment (params, metrics, time, hardware). A model version tells you what can be deployed. Most runs are never deployed. The few that are need things a run doesn't have: promotion, rollback, retirement. Keeping them separate also means that if I rename an experiment or delete old runs, `models:/food11@champion` still works.

**Why aliases are better than stages:** the old stages were a fixed list of 4 values (`None`, `Staging`, `Production`, `Archived`). Every team had to use those names, and you couldn't express things like a canary next to the stable version, or one model per region. Aliases:
- can have any name, and as many as you want
- can be moved from one version to another in one step, and the version numbers don't change, so the history stays clean
- several aliases can point to the same version (`champion` and `eu-prod` at the same time)
- the container only knows the alias, so changing the model is just a change in the registry: no rebuild, no code change

It's the same idea as `data.dvc` in lab 1: a small pointer in front of a big file, and you change the pointer.

## Question 3 - Why load through an MLflow model URI instead of the `.pth` file? What would you change to serve a newer version?

1. **The `.pth` path only exists on my laptop.** That's exactly the problem I had at the start of this lab. `mlruns/1/models/m-.../artifacts/data/model.pth` doesn't exist in the container.
2. **A `.pth` file is only the weights.** It doesn't say what the architecture is, what the input shape is, how to preprocess, or which packages are needed. To load it I'd have to rebuild `resnet18` with an 11-class head in `serve.py`, copying code from `train.py`. The MLflow model folder has all that: `MLmodel` (flavors and signature), `requirements.txt`, `python_env.yaml` and an input example.
3. **`pyfunc` works the same for any model.** `model.predict(array)` works whether the model is PyTorch, scikit-learn or XGBoost, so `serve.py` doesn't need to change if the model type changes.
4. **The alias separates deployment from versions.** The image doesn't need to know which version it will serve.

**To serve a newer version:** move the alias to it.

```python
client.set_registered_model_alias("food11", "champion", 2)
```

Then restart the container. No rebuild, no code change, because the model is loaded at startup. To use a fixed version instead of the alias, I can set `MODEL_URI=models:/food11/2` with `docker run -e`, because `serve.py` reads `MODEL_URI` from the environment.

## Question 4 - Why copy `pyproject.toml` / `uv.lock` and run `uv sync` before copying the source? What happens to the cache when you change one line in `serve.py`?

Docker builds the image layer by layer and **caches** each layer. A layer is reused only if its instruction and files didn't change **and** all the layers before it were reused. So we put what changes least first.

Dependencies change rarely (only when I add a package). The code changes all the time. So the Dockerfile does:

```dockerfile
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project   # GBs of packages, takes minutes
COPY src/ ./src/                                      # a few kB, instant
```

`--no-install-project` installs the dependencies without installing my project, so this layer doesn't need `src/` at all. I checked: with only `pyproject.toml` and `uv.lock` in the folder, this command builds a full venv and `import mlflow, fastapi, torch, PIL` works.

**If I change one line in `serve.py`**, only the `COPY src/` layer and the ones after it are rebuilt. The base image and the whole dependency install come from the cache, so the rebuild takes a few seconds.

If I did `COPY . .` first, any change to any file would make Docker reinstall all the packages on every build.

## Question 5 - Size of a naive single-stage image vs the multi-stage one

Measured with `docker images`:

| Image | Base | Stages | Size | Build context | Build time |
|---|---|---|---|---|---|
| `food11-api:naive` | `python:3.12` | 1 | **11.7 GB** | 4.98 GB | ~45 min |
| `food11-api:latest` | `python:3.12-slim` | 2 | **2.38 GB** | 464 B | ~20 min |

The multi-stage image is about **5 times smaller**.

`docker history food11-api:latest`:

| Layer | Size |
|---|---|
| `COPY --from=builder /app/.venv /app/.venv` | **1.73 GB** |
| Debian base | 87.7 MB |
| Python 3.12 | 41.4 MB |
| apt packages | 4.95 MB |
| `COPY src/ ./src/` | 73.7 kB |
| `RUN useradd` | 69.6 kB |

Almost all of the image is the virtual environment. My code is only 73.7 kB. So if you want a smaller image, you have to work on the dependencies, not on the code.

Three things make the difference, and only the last one comes from multi-stage:
1. **Base image:** `python:3.12` has compilers, git and dev headers. `python:3.12-slim` doesn't (about 800 MB vs 130 MB).
2. **What gets copied:** the naive build does `COPY . .`, the real one only copies `src/`.
3. **The builder stage is thrown away:** `uv`, its cache and temporary files stay in the builder stage and never reach the final image. It's also safer: there's no compiler in the production container that an attacker could use.

The CPU/CUDA change from the top of this report saves even more than these three: about 3 GB of NVIDIA libraries that a container without a GPU never uses.

(The 4.98 GB build context of the naive build was caused by a mistake in my `.dockerignore`, see Q6.)

## Question 6 - What happens without a `.dockerignore`? Which excluded folder would actually break the build?

**Build speed:** before running any instruction, Docker packs the whole folder (the build context) and sends it to the daemon. Without `.dockerignore`, that includes `data_local/` (the full dataset, about 1 GB), `data/`, `.dvc/cache/` (a second copy of all the data), `mlruns/` and `mlflow.db`, `.git/` and `.venv/`. That's several GB sent on **every build**, before the cache can even help.

**Image size:** it only gets bigger if the Dockerfile does `COPY . .` like the naive one. With `COPY src/` only, the files are sent but not copied into the image, so it costs time, not space.

**The one that breaks the build: `.venv/`.** The others only make it slower or bigger. My `.venv` was created on Windows: it has `Scripts\` instead of `bin/`, `.exe` files, and paths like `C:\Users\User\OneDrive\...\.venv`. With `COPY . .` it lands at `/app/.venv`, exactly where the Linux venv should be. If it's still there at runtime, `PATH="/app/.venv/bin:$PATH"` points to a folder without `bin/`, and the container fails with `uvicorn: not found`. The error doesn't say anything about the real cause.

`mlruns/` is the second worst. It wouldn't break the build, but it would put an old copy of the model in the image, which goes against the idea of loading `models:/food11@champion` at runtime.

**A mistake I found in my own `.dockerignore`:** one line had been broken by a formatter. It read `.venv/ \*_/**pycache**/` on a single line. Docker reads each line as one pattern, so that line matched nothing, and **`.venv/`, `__pycache__/` and `*.pyc` were not excluded**. That explains the 4.98 GB build context of the naive build in Q5: my Windows `.venv` (with CUDA torch) was sent on every build.

It didn't break the naive image only because `uv sync` runs after `COPY . .`. uv saw that the copied `.venv` was not a valid Linux environment and rebuilt it. That's why, in the listing below, `.venv` is 14 minutes newer than the `COPY` (15:38 vs 15:24):

```
$ docker run --rm --entrypoint sh food11-api:naive -c "ls -la /app"
drwxr-xr-x  .            Sep 22 15:24
-rwxr-xr-x  .gitignore   Sep 16 22:16
-rwxr-xr-x  .python-version
drwxr-xr-x  .venv        Sep 22 15:38
-rwxr-xr-x  README.md
-rwxr-xr-x  data.dvc
-rwxr-xr-x  pyproject.toml
drwxr-xr-x  src
-rwxr-xr-x  uv.lock
```

The other folders (`data/`, `data_local/`, `mlruns/`, `mlflow.db`, `.git/`) were correctly excluded. The real multi-stage image was never affected, because it only copies `pyproject.toml`, `uv.lock` and `src/`.

I fixed the file so each pattern is on its own line:

```
.venv/
__pycache__/
*.py[cod]
```

## Question 7 - Why can't the container use `127.0.0.1:5000`? What does `host.docker.internal` resolve to?

Each container has its **own network namespace**: its own loopback, its own ports. Inside the container, `127.0.0.1` means the container itself, not my laptop. So `http://127.0.0.1:5000` looks for something listening on port 5000 inside the container. There's nothing, so the connection is refused.

**`host.docker.internal`** is a name that Docker Desktop adds inside containers. It resolves to the address where the host can be reached from the container. On Windows and Mac, Docker Desktop runs containers in a Linux VM, so it's the gateway of that VM, which forwards to the host. It exists because there's no fixed IP for "the machine I'm running on".

- **Windows / Mac:** works directly with Docker Desktop.
- **Linux:** not defined by default. You need `--add-host=host.docker.internal:host-gateway`, or `--network host`.

In my case, fixing the name was not enough. Three things had to be right, and each one failed in a different way:

1. **Name:** `127.0.0.1` means the container itself, so connection refused. Fix: `host.docker.internal`.
2. **Binding:** the server only listened on loopback, so connection refused again. Fix: `--host 0.0.0.0`.
3. **MLflow's check:** the `Host:` header wasn't in the allowed list, so I got a **403** (not a timeout). Fix: `--allowed-hosts`.

The third one surprised me. The first two give connection errors, so you look at the network. The third is a clean 403: the server received the request and refused it on purpose. At that point the container was reaching the server fine, it was just being turned away.

## Question 8 - Stop the container and start a new one. Does the model still load without rebuilding? What does it tell you?

**Yes.** A new container from the same image loads the model again. The startup log shows it every time:

```
INFO:food11.serve:tracking uri: http://host.docker.internal:5000
INFO:food11.serve:loading model: models:/food11@champion
INFO:food11.serve:model loaded
```

MLflow also printed a warning:

```
WARNING mlflow.pyfunc: The version of Python that the model was saved in,
`Python 3.10.11`, differs from the version of Python that is currently running,
`Python 3.12.14`, and may be incompatible
```

I train with Python 3.10 on Windows (from `.python-version`), but the image uses `python:3.12-slim`. The model loaded fine and gave the same predictions as locally, so I kept it like that. A cleaner fix would be to build from `python:3.10-slim` so serving uses the same Python as training, because a pickle saved with one Python version isn't guaranteed to work with another. MLflow could warn me only because `log_model` saved `python_env.yaml` with the model.

**What is inside the image:** the OS, Python, the venv with all the packages, my code, and the **name** of the model to load (`models:/food11@champion`), just a string, not the weights.

**What is fetched when the container starts:** which version the alias points to, and that version's files, downloaded from the tracking server.

So the image doesn't contain the model. The same image can serve version 1 today and version 7 later. Changing the model is a change in the registry, not a new build, and CI can build the image before anyone decides which model is champion.

The downside: the container **depends on the tracking server**. If MLflow is down when the container starts, it can't serve anything, so a problem in the experiment tracking becomes a problem in production. Startup is also slower because about 45 MB has to be downloaded first. The other option is to put the model in the image at build time: no runtime dependency, but you need a rebuild for every new model. Both are used in real systems.

## Question 9 - What is still missing before another machine can pull and run this exact image?

The Dockerfile is in git, but the image only exists on my laptop. A CI runner or a Kubernetes node can't get it. What's missing:

1. **A registry:** the image has to be pushed somewhere others can pull from (Docker Hub, GHCR, ECR) with `docker tag` and `docker push`.
2. **A fixed tag:** `latest` can point to a different image tomorrow, so two machines pulling it at different times can get different images. You should use a version tag, or better, the digest (`@sha256:...`), which always means the same image.
3. **A link to the code:** nothing in the image says which commit built it. A label like `org.opencontainers.image.revision=$GIT_SHA` would let me find the code from a running container.
4. **A tracking server and artifact store everyone can reach:** `host.docker.internal` means nothing in a cluster, and `./mlartifacts` on my laptop can't be reached from elsewhere. Production needs a real MLflow server plus object storage (S3, GCS, DagsHub), with credentials passed as secrets.
5. **Deployment config:** Kubernetes needs manifests with resource limits and liveness/readiness probes. `/health` exists for that, but nothing uses it yet. The readiness probe is what would stop traffic going to a container whose model failed to load.
6. **An automatic build:** if I build by hand on my laptop, the image depends on my laptop's state. CI building from a clean checkout on every push is what makes "the exact image" reliable.

Points 1, 2, 3 and 6 are about CI/CD, and points 4 and 5 are about Kubernetes, which are the next labs.

---

## Files I made in this lab

- `src/food11/serve.py`: the FastAPI service. It loads `models:/food11@champion` once at startup through the tracking server, and has `GET /health` and `POST /predict`.
- `Dockerfile`: two-stage build. `uv sync` builds the venv in the builder stage, then a slim runtime stage gets only the venv and `src/`.
- `Dockerfile.naive`: a bad single-stage build, only kept to compare sizes in Q5.
- `.dockerignore`: keeps the data, the dvc cache, the MLflow files, `.git` and the Windows `.venv` out of the build context (fixed, see Q6).
- `lab/lab3.md`: this report.

### Notes on the code

- **Class names are hardcoded in `serve.py`, in sorted order.** The model only outputs 11 numbers, no names. `ImageFolder` gives each class an index by sorting the folder names, so the list in `serve.py` must stay sorted to match training. A better way would be to log the class list with the model and read it back.
- **Preprocessing is written twice**, in `train.py` and `serve.py` (resize to 128, ImageNet normalization). If one changes and not the other, the model gets inputs that don't look like its training data, and accuracy drops without any error (training/serving skew). The signature (`[-1, 3, 128, 128]` float32) catches a wrong shape, but not a wrong normalization.
- **I tested the service, I didn't just assume it works.** My first test (a Bread image) came back as "Fried food" with 0.76 confidence. That could be a normal mistake of a 75% model, or a bug in the class order, and one image can't tell the difference. So I sent 5 validation images per category to the API and compared with the folder names:

  ```
  served accuracy: 41 / 55   (74.5%)
  ```

  That's very close to the 0.7482 `val_accuracy` from training, so the service reproduces the model, and the Bread image was just a wrong prediction. If the class order were wrong I'd get around 9% (random), and with wrong normalization something in between. The container gave 0.7607 for an image that gave 0.7605 locally, a tiny difference between the CPU and CUDA versions of torch.
- **The model is loaded at startup, not on each request.** If `MODEL_URI` is wrong, the container fails right away instead of on the first user request, and the ~45 MB download only happens once.
