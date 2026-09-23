# Lab 3 — Containerizing the model with Docker

**Repository:** https://github.com/englaetitia/mlops-lab-1
**Author:** Laetitia Daou

## Three changes needed before the lab could work

### 1. The tracking server had to start serving artifacts

In Lab 2 the server was started with `--default-artifact-root ./mlruns`, which records
each experiment's artifact location as an **absolute path on the host filesystem**:

```
(1, 'food11', 'C:\...\mlops-lab-1\mlruns\1')
```

A client resolving `models:/food11@champion` then tries to open that path *directly*.
That works on the laptop that created it and fails everywhere else — including inside a
container, which has no such directory. The Dockerfile is irrelevant to this; the model
simply cannot be fetched.

The fix is to let the tracking server proxy artifacts over HTTP:

```bash
mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --artifacts-destination ./mlartifacts --serve-artifacts
```

Experiments created under this configuration record a **server-relative** URI instead:

```
(2, 'food11', 'mlflow-artifacts:/2')
```

which any client resolves by asking the server for the bytes. Existing experiments keep
their old absolute paths, so the Lab 2 `food11` experiment was renamed to `food11-lab2`
and the winning configuration re-run to create a fresh one.

`--host` also had to change from `127.0.0.1` to `0.0.0.0`. Bound to loopback the server
accepts connections only from the host itself, so the container's request would be
refused even with the networking in Q7 set up correctly.

### 2. The server had to allow the container's `Host` header

With networking working and the artifact store fixed, the container still failed:

```
MlflowException: API request to endpoint /api/2.0/mlflow/registered-models/alias
failed with error code 403 != 200.
Response body: 'Invalid Host header - possible DNS rebinding attack detected'
```

MLflow validates the `Host:` header on every request to prevent DNS rebinding attacks.
Its default allow-list is localhost, the loopback addresses and RFC-1918 private ranges.
The container sends `Host: host.docker.internal:5000` — a hostname matching none of those —
so the request was rejected before reaching any handler. The connection itself was fine;
this is an application-layer refusal, which is why it produced a clean 403 rather than a
timeout.

```bash
mlflow server ... --allowed-hosts "localhost,localhost:*,127.0.0.1,127.0.0.1:*,host.docker.internal,host.docker.internal:*"
```

`--allowed-hosts` **replaces** the defaults rather than extending them, so the localhost
entries have to be repeated or the browser UI and local scripts start getting 403s
instead. Verified both directions: with this list, a request carrying
`Host: host.docker.internal:5000` returns 200, while an unlisted host still returns 403 —
the protection stays on, it just knows about one more legitimate name.

### 3. The container gets CPU-only PyTorch

`uv.lock` pinned the CUDA build of torch (~3 GB of NVIDIA runtime libraries) because the
laptop has an RTX 3050. A serving container has no GPU, so those gigabytes are dead
weight. `[tool.uv.sources]` accepts environment markers, which lets one lock file serve
both:

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

Windows training keeps CUDA; the Linux image gets the CPU wheels. Same lock file, same
`uv sync --frozen`.

---

## Question 1 — What version number was your model given? What's the difference between a run's logged model artifact and a registered model?

**Version 1.**

| | Logged model artifact | Registered model |
|---|---|---|
| Identity | Belongs to one run; addressed as `runs:/<run_id>/model` | A named entry (`food11`) with its own version sequence |
| Lifetime | Tied to the experiment that produced it | Independent of experiments and runs |
| Naming | Whatever run happened to produce it | A stable, human-chosen name |
| Metadata | Flavors, signature, environment | Adds versions, aliases, tags, descriptions, history |
| Referenced as | `runs:/36d97ad.../model` | `models:/food11@champion` |

**Registering does not copy anything.** The new version's source came back as
`models:/m-cdc819e8c15e4a0cbc019d764ce11844` — a pointer to the same logged model. The
registry is a naming and governance layer over artifacts that already exist.

The practical difference is who has to know what. A run ID is an accident of history: to
deploy by run ID, the serving code must be told which of dozens of runs won, and that
changes every time you retrain. `models:/food11@champion` never changes. The registry is
the seam between *experimentation*, where runs are cheap and disposable, and *operations*,
where one artifact is the one in production.

## Question 2 — What aliases replaced the old stages? Why version separately from the run, and why is an alias more flexible than a stage?

**Aliases** replaced the fixed stage enum. They are free-form names — the conventional
ones being `champion` for what is live and `challenger` for what is being evaluated
against it — but any string works: `canary`, `eu-prod`, `rollback-target`.

**Why version separately from the run.** A run records *what happened during an
experiment*: hyperparameters, metric curves, duration, hardware. A model version records
*what is deployable*. Most runs never produce a model anyone ships; a few do, and those
need a lifecycle the run has no vocabulary for — promotion, rollback, deprecation.
Keeping them separate also means the deployment surface survives changes to how
experiments are organised: rename an experiment, delete old runs, switch tracking
backends, and `models:/food11@champion` still resolves.

**Why aliases beat stages.** The old stages were a closed set of four values
(`None`, `Staging`, `Production`, `Archived`) baked into MLflow. That forces every
team's release process into one vocabulary, and there was no way to express two things at
once — a canary alongside a stable version, or different models per region.

Aliases fix this by being data rather than schema:

- **Arbitrary and unlimited.** Any name, as many as you like.
- **Reassignable atomically.** Promotion is repointing a pointer; the version numbers
  never change, so the audit trail stays intact.
- **Many-to-one.** One version can hold `champion` and `eu-prod` simultaneously.
- **Deployment-config friendly.** The container references the alias, so promoting a new
  model is a registry operation — no rebuild, no redeploy, no code change.

The underlying idea is the one behind Lab 1's `data.dvc`: put a stable, cheap pointer in
front of an expensive immutable object, and do your version management on the pointer.

## Question 3 — Why load through an mlflow model URI instead of the `.pth` file? What would you change to serve a newer version?

Four reasons, in rough order of how badly each bites:

1. **The path doesn't exist anywhere else.** This is not hypothetical — it is exactly the
   failure described at the top of this report. `mlruns/1/models/m-.../artifacts/data/model.pth`
   is a path on one Windows laptop. The container's filesystem has no such directory.
2. **A `.pth` is only weights.** It carries no architecture, no input shape, no
   preprocessing contract, no dependency list. Loading it requires the serving code to
   reconstruct `resnet18`, know that the head is 11-way, and know the images are
   128×128 normalised with ImageNet statistics — all duplicated from `train.py` and all
   free to drift. The MLflow model directory carries `MLmodel` (flavors and signature),
   `requirements.txt`, `python_env.yaml` and an input example alongside the weights.
3. **`pyfunc` is a uniform interface.** `model.predict(array)` works the same whether the
   underlying model is PyTorch, scikit-learn or XGBoost. Swapping the model later doesn't
   touch `serve.py`.
4. **The alias decouples deployment from versioning.** The image is built without knowing
   which model version it will serve.

**To serve a newer version:** reassign the alias.

```python
client.set_registered_model_alias("food11", "champion", 2)
```

Then restart the container — nothing is rebuilt and no code changes, because the model is
resolved at startup rather than baked in. To pin a specific version instead of following
the alias, set `MODEL_URI=models:/food11/2`; `serve.py` reads that from the environment,
so it is a `docker run -e` flag, not an edit.

## Question 4 — Why copy `pyproject.toml`/`uv.lock` and run `uv sync` before the source? What happens to the cache when you change a line in `serve.py`?

Docker builds an image as a stack of layers, and caches each one against a key derived
from the instruction and the content of the files it touches. A layer is reused only if
its key and every key before it are unchanged. So the ordering rule is: **least
frequently changed first.**

Dependencies change rarely — only when you add a package. Source changes constantly. The
Dockerfile therefore does:

```dockerfile
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project   # ~3 GB of wheels, minutes
COPY src/ ./src/                                      # a few kB, instant
```

`--no-install-project` is what makes this possible: it installs the dependencies without
building the project itself, so the install layer has no dependency on `src/` at all.
(Verified: with only `pyproject.toml` and `uv.lock` in the directory, `uv sync --frozen
--no-dev --no-install-project` builds a complete virtualenv and `import mlflow, fastapi,
torch, PIL` all succeed.)

**Changing a line in `serve.py`** invalidates the `COPY src/` layer and everything after
it — which is one small copy and the metadata instructions. The base image pull and the
entire dependency install are reused from cache, so the rebuild takes seconds.

Reverse the order and every rebuild re-downloads and reinstalls several gigabytes,
because `COPY . .` sees a changed file and invalidates the `RUN uv sync` that follows it.

## Question 5 — Size difference between a naive single-stage image and the multi-stage one

Measured with `docker images` after building both:

| Image | Base | Stages | Size | Build context | Build time |
|---|---|---|---|---|---|
| `food11-api:naive` | `python:3.12` | 1 | **11.7 GB** | 4.98 GB | ~45 min |
| `food11-api:latest` | `python:3.12-slim` | 2 | **2.38 GB** | 464 B | ~20 min |

Roughly a **5× difference**.

`docker history food11-api:latest` shows where the bulk sits:

| Layer | Size |
|---|---|
| `COPY --from=builder /app/.venv /app/.venv` | **1.73 GB** |
| Debian trixie base | 87.7 MB |
| Python 3.12.14 install | 41.4 MB |
| apt dependencies | 4.95 MB |
| `COPY src/ ./src/` | 73.7 kB |
| `RUN useradd` | 69.6 kB |

The virtualenv is the image. Application code is 73.7 kB — 0.004% of the total. Every
byte of optimisation that matters is in dependency management, not in how the source is
packaged, which is a useful thing to internalise before trying to shave layers.

Three separate savings are in play, and only the last is specific to multi-stage:

1. **Base image.** `python:3.12` ships a compiler toolchain, git and dev headers;
   `python:3.12-slim` does not. Roughly 800 MB versus 130 MB before anything is installed.
2. **Build context.** The naive build's `COPY . .` puts the whole repository into the
   image; the real one copies only `src/`.
3. **Discarded build stage.** `uv` itself, the uv cache and any intermediate wheels exist
   only in the builder stage and never reach the final image. Nothing in the runtime
   stage can build software — which is also a security improvement, since a compiler in a
   production container is a tool for an attacker.

The CPU/CUDA split described at the top of this report saves more than all three
combined: about 3 GB of NVIDIA runtime libraries that a GPU-less container would never
load.

## Question 6 — What happens without a `.dockerignore`? Which excluded folder would actually break the build?

**Build speed.** Before executing a single instruction, the Docker CLI packs the entire
build context into a tarball and sends it to the daemon. Without a `.dockerignore` that
context includes `data_local/` (the full ~1 GB Food-11 download), `data/` (the processed
datasets), `.dvc/cache/` (a second complete copy of everything DVC tracks), `mlruns/` and
`mlflow.db` (model artifacts, ~45 MB per run), `.git/`, and `.venv/`. Several gigabytes,
packed and transferred **on every build**, before any caching can help.

**Image size.** Only if the Dockerfile uses `COPY . .`, as the naive one does — then all
of it lands in the image too. With an explicit `COPY src/`, the context is still sent but
not copied in, so the cost is speed rather than size.

**Which one actually breaks the build: `.venv/`.** The others bloat; this one is
poisonous. The virtualenv was built on Windows, so it contains `Scripts\` rather than
`bin/`, `.exe` launchers, and absolute paths pointing at
`C:\Users\User\OneDrive\...\.venv`. Copied into a Linux image with `COPY . .`, it would
land at `/app/.venv` — exactly where the Dockerfile puts the *correct* Linux virtualenv —
and shadow it. `ENV PATH="/app/.venv/bin:$PATH"` would then point at a directory with no
`bin/`, and the container would fail at startup with `uvicorn: not found`, with nothing in
the error pointing at the real cause.

`mlruns/` is the runner-up: it wouldn't break the build, but it would bake a stale copy of
the model into the image, quietly defeating the whole point of resolving
`models:/food11@champion` at runtime.

**Verified that the `.dockerignore` is in fact doing its job.** Listing `/app` inside the
naive image — the one built with `COPY . .`, where anything not excluded would be visible:

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

No `data/`, no `data_local/`, no `mlruns/`, no `mlflow.db`, no `.git/` — all correctly
excluded. The `.venv` present is *not* the Windows one: `/app` is timestamped 15:24 (the
`COPY . .`) while `.venv` is 15:38, fourteen minutes later, which is when `uv sync`
finished building it inside the container. So the shadowing failure above is a hazard the
`.dockerignore` prevented, not one that was observed.

One number remains unexplained: the naive build reported `transferring context: 4.98GB`
even though the image contents show the large directories were excluded. The most likely
cause is Docker Desktop for Windows reporting bytes scanned across its file-sharing layer
rather than bytes actually transferred. It was not investigated further, since the image
contents settle the question the lab is asking.

## Question 7 — Why can't the container use `127.0.0.1:5000`? What does `host.docker.internal` resolve to?

Every container gets its **own network namespace**: its own loopback interface, its own
routing table, its own set of ports. `127.0.0.1` inside the container means *this
container*, not the machine running it. So `http://127.0.0.1:5000` asks whether anything
in the container is listening on port 5000 — nothing is, and the connection is refused.
The MLflow server is a completely separate namespace away.

**`host.docker.internal`** is a DNS name Docker Desktop injects into containers, which
resolves to the address at which the host is reachable from inside the container. On
Windows and macOS, Docker Desktop runs containers inside a Linux VM, so it resolves to
that VM's gateway address, which forwards to the host. It exists precisely because there
is no portable, stable IP for "the machine I'm running on".

Platform differences worth knowing:

- **Windows/macOS:** works out of the box with Docker Desktop.
- **Linux:** the name is not defined by default; you need
  `--add-host=host.docker.internal:host-gateway`, or `--network host` to skip the
  namespace isolation entirely.

**Resolving the name was necessary but not sufficient.** Three separate things had to be
true before the container could talk to the tracking server, and each failed differently:

| Layer | Problem | Symptom | Fix |
|---|---|---|---|
| Naming | `127.0.0.1` means the container itself | connection refused | `host.docker.internal` |
| Binding | server listening only on loopback | connection refused | `--host 0.0.0.0` |
| Application | `Host:` header not in MLflow's allow-list | **403**, not a timeout | `--allowed-hosts` |

The third was the instructive one. The first two produce connection errors that point
roughly at the network; the third produces a clean HTTP 403 from a server that received
the request, parsed it, and deliberately rejected it. "Container can't reach the server"
was the wrong mental model at that point — the container reached it perfectly well and was
turned away at the door.

## Question 8 — Does the model still load in a fresh container without rebuilding? What does that tell you?

**Yes.** Stopping the container and starting a new one from the same image works, and the
model loads again. The startup log shows it happening every time:

```
INFO:food11.serve:tracking uri: http://host.docker.internal:5000
INFO:food11.serve:loading model: models:/food11@champion
INFO:food11.serve:model loaded
```

MLflow also flags a mismatch it noticed while loading:

```
WARNING mlflow.pyfunc: The version of Python that the model was saved in,
`Python 3.10.11`, differs from the version of Python that is currently running,
`Python 3.12.14`, and may be incompatible
```

Training runs on the Windows `.venv` (Python 3.10, pinned by `.python-version`); the image
is built `FROM python:3.12-slim`. The model unpickled correctly and produced output
identical to the local run, so nothing broke — but this is the same class of problem as the
torch version, one level up, and MLflow only knows to warn because `log_model` recorded
`python_env.yaml` alongside the weights. The correct fix is to build the image from
`python:3.10-slim` so the serving interpreter matches the one that produced the pickle.
Pickle compatibility across Python minor versions is a convention, not a guarantee.

**Baked into the image:** the OS layer, Python, the virtualenv with every dependency, the
source code, and the *identity* of the model to serve (`models:/food11@champion`) — a
string, not bytes.

**Fetched at runtime:** the alias is resolved to a concrete version, and that version's
artifacts are downloaded from the tracking server, every time a container starts.

What this buys is that the image is **stateless with respect to the model**. The same
image serves version 1 today and version 7 next month; promoting a model is a registry
write, not a build. It also means the image can be built by CI long before anyone decides
which model is champion.

What it costs is equally worth stating: the container now has a **runtime dependency on
the tracking server**. If MLflow is unreachable at startup, the container cannot serve at
all — an outage in the experiment-tracking system becomes an outage in production
inference. Startup is also slower, since ~45 MB is downloaded before the first request can
be answered. The alternative — baking the model in at build time — trades that
availability risk for an immutable, self-contained image that needs a rebuild for every
promotion. Neither is universally right; real systems often bake the model in and treat
the registry as the build-time source of truth.

## Question 9 — What's still missing before another machine could pull and run this exact image?

The Dockerfile is versioned in git; the image exists only in the local Docker daemon. A CI
runner or Kubernetes node cannot reach it. Six things are missing:

1. **A registry.** The image has to be pushed somewhere reachable — Docker Hub, GHCR,
   ECR — via `docker tag` and `docker push`. Without this there is nothing to pull.
2. **An immutable tag.** `latest` is a mutable pointer; two machines pulling it a week
   apart can get different images. Deployments should reference a version tag
   (`food11-api:1.0.0`) or, better, the content digest (`@sha256:...`), which is the only
   truly reproducible reference.
3. **Provenance.** Nothing in the image records which commit built it. OCI labels
   (`org.opencontainers.image.revision=$GIT_SHA`) would make it possible to work backwards
   from a running container to the source.
4. **A reachable tracking server and a shared artifact store.** `host.docker.internal`
   means nothing in a cluster, and `./mlartifacts` on a laptop is not reachable from
   anywhere. Production needs a real MLflow endpoint plus object storage (S3, GCS,
   DagsHub), with credentials injected as secrets rather than environment defaults.
5. **Deployment configuration.** Kubernetes needs manifests declaring resource requests
   and limits, and liveness/readiness probes. `/health` exists for exactly this, but
   nothing currently declares it as a probe — and since the model loads at startup, the
   readiness probe is what would stop traffic reaching a container whose model fetch
   failed.
6. **The build itself has to be automated.** Building by hand on a laptop means the image
   depends on one machine's state. CI building from a clean checkout on every push is
   what makes "the exact image" a meaningful phrase.

Items 1–3 and 6 are the substance of CI/CD; items 4–5 are the substance of the Kubernetes
lab. They are the next two labs.

---

## Files produced in this lab

| File | Purpose |
|---|---|
| `src/food11/serve.py` | FastAPI service. Loads `models:/food11@champion` once at startup through the tracking server; `GET /health`, `POST /predict`. |
| `Dockerfile` | Two-stage build: `uv sync` into a virtualenv in the builder, then a slim runtime carrying only the venv and `src/`. |
| `Dockerfile.naive` | Deliberately bad single-stage build, kept only to produce the comparison in Q5. |
| `.dockerignore` | Keeps the datasets, DVC cache, mlflow store, `.git` and the Windows `.venv` out of the build context. |
| `lab/lab3.md` | This file. |

### Implementation notes

**Class names are hardcoded in `serve.py` and sorted.** The model outputs 11 logits and
carries no label names. `torchvision.datasets.ImageFolder` assigns indices by sorting the
class directory names, so the list in `serve.py` must stay in sorted order to match what
training used. A more robust design would log the class list as a run parameter and read
it back with the model — worth doing before this ever matters in production.

**Preprocessing is duplicated between `train.py` and `serve.py`** (resize to 128,
normalise with ImageNet statistics). This is the classic training/serving skew risk: if
one changes and the other doesn't, the model receives inputs unlike anything it was
trained on and accuracy degrades silently, with no error anywhere. The logged model's
signature (`[-1, 3, 128, 128]` float32) catches shape mismatches but not a wrong
normalisation.

**So the service was measured, not assumed.** The first prediction tried — a Bread image —
came back as "Fried food" at 0.76 confidence, which is equally consistent with a correct
service making one of its 25% errors and with a broken class mapping. One sample cannot
distinguish them. Running 5 validation images per category through the API and checking
each answer against its folder name gave:

```
served accuracy: 41 / 55   (74.5%)
```

against `val_accuracy` 0.7482 logged during training. The service reproduces the model, so
the Bread miss was a genuine misclassification. Had the class list been mis-ordered, this
would have come out near 9% (1/11, chance); had normalisation been wrong, somewhere in
between. `val_accuracy` measures the model; this measures the *service*, and they are only
equal if `serve.py` reproduces the training transforms exactly.

The containerised service was checked the same way and returned 0.7607 for the image that
scored 0.7605 locally — a floating-point difference between the CPU and CUDA torch builds,
not a behavioural one.

**The model is loaded in a startup handler, not per request.** A bad `MODEL_URI` then
fails loudly when the container boots rather than on a user's first request, and the
~45 MB download happens once.
