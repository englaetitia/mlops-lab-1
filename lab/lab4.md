# Lab 4 — Orchestrating the app with Docker Compose

**Repository:** https://github.com/englaetitia/mlops-lab-1
**Author:** Laetitia Daou

## Deviations from the lab sheet

Three, each for a reason that would otherwise have stopped the stack working.

### 1. `--artifacts-destination` + `--serve-artifacts`, not `--default-artifact-root`

The lab's `mlflow/Dockerfile` uses `--default-artifact-root /mlflow-data/mlruns`. That
records each experiment's artifact location as an **absolute path inside the mlflow
container**. The inference container has no `/mlflow-data`, so `mlflow.pyfunc.load_model`
would try to open a directory that does not exist there. This is the same failure as in
Lab 3, where the path was a Windows folder instead of a container folder — the lesson
generalises: *an artifact location that is a filesystem path only works for clients that
share that filesystem.*

With `--serve-artifacts`, locations are recorded as `mlflow-artifacts:/<experiment_id>`
and every client fetches the bytes over HTTP from the tracking server, which is the only
process that needs the volume mounted.

### 2. `--allowed-hosts` must include the service name

MLflow validates the HTTP `Host:` header to block DNS-rebinding attacks, and its default
allow-list is localhost plus RFC-1918 private IP ranges. The inference container reaches
the server as `http://mlflow:5000`, sending `Host: mlflow:5000` — a hostname matching
nothing in that list — so every request would be refused with **403 Invalid Host header**.
The same flag was needed in Lab 3 for `host.docker.internal`.

```
--allowed-hosts "localhost,localhost:*,127.0.0.1,127.0.0.1:*,mlflow,mlflow:*"
```

The flag replaces the defaults rather than extending them, so the localhost entries have
to be repeated or the browser UI on `127.0.0.1:5000` stops working too.

### 3. Aliases, not `Staging`

The lab sheet asks for the model to be moved to `Staging`. Registry stages are deprecated
in current MLflow — the Lab 3 sheet says so itself — so this lab uses the `champion`
alias established in Lab 3. Everything the questions ask about promotion works identically;
only the vocabulary differs.

### A fourth, smaller one: reusing the Lab 3 image

`docker-compose.yml` gives the inference service both `build: .` and
`image: food11-api:latest`, so Compose uses the image already built in Lab 3 instead of
rebuilding it. This mattered in practice: Compose builds services **in parallel**, and
building `frontend` and `inference` simultaneously split the available bandwidth between
them. Transfer rates dropped to ~50 kB/s and `uv` aborted with

```
error: Failed to download `numpy==2.5.3`
  cause: Failed to download distribution due to network timeout (current value: 30s)
```

Not a broken Dockerfile — two builds starving each other. Building one service at a time
fixed it, and `UV_HTTP_TIMEOUT=300` was added to the builder stage so a slow link fails
slowly rather than fatally.

---

## Question 1 — What happens to `/mlflow-data` with no volume mounted?

Everything written there goes into the **container's writable layer** and is destroyed
with the container. Tested directly, running the image standalone with no `-v`:

```
> docker run --rm -d --name novol -p 5001:5000 mlops-lab-1-mlflow
> docker exec novol python -c "... mlflow.set_experiment('will-i-survive')"
INFO mlflow.tracking.fluent: Experiment with name 'will-i-survive' does not exist. Creating a new experiment.
created

> docker stop novol
> docker run --rm -d --name novol -p 5001:5000 mlops-lab-1-mlflow
> docker exec novol python -c "... print([e.name for e in mlflow.search_experiments()])"
['Default']
```

`will-i-survive` is gone. The second container starts from the image, which contains an
empty `/mlflow-data`; the SQLite file the first container wrote was never part of the
image and disappeared with `--rm`.

The same thing was observed accidentally later in the lab: a test image copied into the
inference container with `docker compose cp` vanished after a `docker compose down`,
because `down` removes containers. Anything not on a volume is scratch space.

## Question 2 — Why a named volume rather than a bind mount?

A bind mount *would* work — the data would persist, and you could open `mlflow.db` in your
editor. Four reasons the named volume is the better default here:

1. **Docker owns the lifecycle.** The volume is created by `docker compose up`, survives
   `down`, and is destroyed only by `down -v`. Its lifetime is tied to the *stack*, not to
   a folder that happens to exist in a checkout.
2. **No host path in the compose file.** A bind mount hard-codes a path that must exist on
   every machine running the stack. `mlflow-data:` is portable; `C:\Users\User\OneDrive\...`
   is not.
3. **No permission or filesystem mismatch.** The mlflow container runs as root on Linux;
   a bind mount to a Windows folder crosses Docker Desktop's file-sharing layer, where
   SQLite's file locking is notoriously unreliable. A named volume is a native Linux
   filesystem — confirmed by `df -h` inside the container showing `/dev/sdd` mounted at
   `/mlflow-data`, distinct from the container's overlay root.
4. **It doesn't end up in git or the build context.** A bind-mounted `./mlflow-data/`
   inside the repo would have to be added to both `.gitignore` and `.dockerignore`, and
   would be one forgotten entry away from a database in a commit.

A bind mount is right for *code you are editing*, where seeing and changing files from the
host is the whole point. It is wrong for *state the application owns*.

## Question 3 — Why does `http://mlflow:5000` resolve now?

Compose creates a private network for the stack and runs an **embedded DNS server** on it.
Every service gets a DNS record under its service name, resolving to that container's IP
on that network. So `mlflow` is a real hostname inside the stack, and
`MLFLOW_TRACKING_URI=http://mlflow:5000` just works:

```
INFO:food11.serve:tracking uri: http://mlflow:5000
INFO:food11.serve:loading model: models:/food11@champion
INFO:food11.serve:model loaded
```

In Lab 3 there was no such network. A single `docker run` container sat on the default
bridge with no knowledge of any other container, so reaching the host's MLflow required
`host.docker.internal` — a name Docker Desktop injects specifically because a container
has no portable way to refer to its host.

The DNS is **dynamic**, which showed up clearly when mlflow was stopped deliberately:

```
NameResolutionError: HTTPConnection(host='mlflow', port=5000):
Failed to resolve 'mlflow' ([Errno -5] No address associated with hostname)
```

Not "connection refused" — the name stopped existing. Compose's DNS only holds records for
running containers, so a stopped service disappears from the namespace entirely rather
than becoming an address that refuses connections.

## Question 4 — Why read `INFERENCE_URL` from the environment?

Because the image should not encode where it is deployed. Hard-coding
`http://inference:8000` would bake in an assumption that only holds inside this particular
Compose stack:

- Running the frontend image standalone against an API on the host would be impossible
  without editing and rebuilding.
- Renaming the service in `docker-compose.yml` would silently break the frontend.
- A Kubernetes deployment, where the service might be
  `http://inference.default.svc.cluster.local:8000`, would need a different image.

With an environment variable there is **one image and many configurations**, the default
(`http://127.0.0.1:8000`) covering local development. This is the same principle as
`MLFLOW_TRACKING_URI` in `serve.py` and `MODEL_URI` in the compose file: anything that
varies between environments is configuration, not code, and configuration does not belong
in a layer.

## Question 5 — Why doesn't `inference` publish a port?

Publishing maps a container port onto a **host** port. It is needed only when something
outside the Compose network has to connect. Two services qualify: `mlflow` (you open its
UI) and `frontend` (you open the page). Nothing outside ever calls the inference API —
only the frontend does, and the frontend is *inside* the network.

The frontend reaches it at `http://inference:8000` through the same embedded DNS as Q3.
The page renders `Model: models:/food11@champion`, which it can only know by having
successfully called `GET http://inference:8000/health` — proof the internal route works
with no published port at all.

Not publishing is a deliberate security posture, not an omission. A published port is
reachable from anything that can reach the host, so every one is a decision to expose
something. The inference API has no authentication; on a shared network, publishing it
would let anyone on that network run inference, upload arbitrary files to it, and read
`/health`. Keeping it internal means the attack surface is exactly the two ports a human
needs.

## Question 6 — `depends_on` waits for start, not readiness

Left as written in the lab sheet, `depends_on: [mlflow]` only guarantees Compose starts
the mlflow container first. The MLflow server inside then needs ~15 seconds for Alembic
migrations and worker startup, during which it accepts nothing. `serve.py` loads the model
at startup, so it would hit a server that is up but not listening, and the container would
exit.

Demonstrated on purpose by stopping mlflow and recreating inference against it:

```
> docker compose stop mlflow
> docker compose up -d --force-recreate --no-deps inference
> docker compose logs inference --tail 30
INFO:food11.serve:tracking uri: http://mlflow:5000
INFO:food11.serve:loading model: models:/food11@champion
WARNING:urllib3.connectionpool: Retrying (Retry(total=6, ...)) after connection broken by
  'NameResolutionError("HTTPConnection(host='mlflow', port=5000): Failed to resolve 'mlflow'")'
WARNING:urllib3.connectionpool: Retrying (Retry(total=5, ...)) ...
```

The fix used here is a **healthcheck** on mlflow plus a conditional dependency:

```yaml
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:5000/health').read()"]
      interval: 5s
      timeout: 3s
      retries: 20
      start_period: 15s
```

```yaml
    depends_on:
      mlflow:
        condition: service_healthy
```

Compose then waits for the server to actually answer before starting inference — visible
in the startup output as `✔ Container m... Healthy 16.9s` before the other two start.
`restart: on-failure` is a second line of defence for cases the healthcheck can't cover,
such as the alias not existing yet.

`/health` is a good probe here because MLflow exempts it from the Host-header validation
described above, so it answers without needing the container to be in `--allowed-hosts`.

## Question 7 — `docker compose ps`

```
NAME                      SERVICE     STATUS             PORTS
mlops-lab-1-mlflow-1      mlflow      Up (healthy)       0.0.0.0:5000->5000/tcp
mlops-lab-1-inference-1   inference   Up                 8000/tcp
mlops-lab-1-frontend-1    frontend    Up                 0.0.0.0:8501->8501/tcp
```

Exactly what the compose file declares. The distinction in that PORTS column is worth
reading carefully:

- `0.0.0.0:5000->5000/tcp` — **published**: a host port is mapped to a container port.
- `8000/tcp` — **exposed only**: the container listens on 8000 and other containers on the
  network can reach it, but no host port maps to it.

`EXPOSE 8000` in the Lab 3 Dockerfile is documentation, not a network operation. It
records which port the image listens on; it does not open anything.

## Question 8 — Does the prediction come from the new version?

**No.** Measured directly through the API, on one fixed Dessert image, changing exactly
one thing at a time:

| # | Registry `champion` | Version in the container | Prediction |
|---|---|---|---|
| 1 | v1 | v1 | Dessert **0.5768** |
| 2 | v2 | v2 (after restart) | Dessert **0.1597** |
| 3 | **v1** | v2 (no restart) | Dessert **0.1597** |
| 4 | v1 | v1 (after restart) | Dessert **0.5768** |

Reading 3 is the answer: the registry said version 1, the service returned version 2's
prediction, unchanged to four decimal places. `mlflow.pyfunc.load_model` resolves the alias
**once**, at startup, and what it returns is an object in memory. Nothing re-reads the
registry afterwards — there is no subscription, no polling, no callback.

The single command that picks up the new version:

```bash
docker compose restart inference
```

(Version 2 was trained deliberately badly — `lr=0.01`, 20.4% test accuracy versus version
1's 77.8% — so the difference would be visible. Note it still got this image's category
*right*, at 16% confidence against version 1's 58%. A correct answer from a bad model is
not evidence of a good model, the same point as the misclassified Bread image in Lab 3.)

## Question 9 — Why does `restart` alone work?

Because nothing about the model is in the image. Restarting re-runs the same container's
entrypoint, which re-executes `serve.py`'s startup handler, which resolves
`models:/food11@champion` afresh and downloads whatever it now points at.

**Baked into the image:** the OS, Python, the virtualenv, `serve.py`, and the *string*
`models:/food11@champion`.

**Fetched at container startup:** the alias resolution, and the model bytes.

So promoting a model is a registry write plus a process restart. No `docker build`, no
registry push, no code change, no redeployment of anything. The image built in Lab 3 —
before version 2 existed — serves version 2 without modification.

The cost is the mirror image of the benefit, and it showed up in Q6: the container cannot
start at all without the tracking server. An outage in experiment tracking becomes an
outage in production inference.

## Question 10 — Persistence across `down` / `up`, and `down -v`

**`docker compose down` — everything survived.** After two full cycles:

```
> docker compose down
> docker compose up -d --no-build
> ... check ...
champion is version 1 | experiments: ['food11', 'Default']
```

`down` stops and removes the containers and the network. The named volume is deliberately
**not** removed, so `mlflow.db` and `mlartifacts/` are still there and the new mlflow
container mounts the same storage.

**`docker compose down -v` — everything gone.** The output says so explicitly:

```
> docker compose down -v
 ✔ Volume mlops... Removed
```

and the proof came on the next training run:

```
INFO mlflow.tracking.fluent: Experiment with name 'food11' does not exist. Creating a new experiment.
```

The experiment, both registered versions, the `champion` alias and every artifact were
destroyed together. The inference service could no longer resolve its model and had
nothing to serve. Recovery required retraining and re-registering from scratch (~40s here,
because the dataset is small and still on the host — with a real training job it would be
hours, or impossible if the data were gone too).

**The difference is one flag**, and it is the difference between "stop the stack" and
"destroy the stack's state". Worth internalising before typing `-v` on anything that
matters.

## Question 11 — What would replicas or a surviving mlflow require?

Docker Compose runs containers **on one machine**. Everything below follows from that.

**For three inference replicas behind a load balancer:**

- `deploy: replicas: 3` exists in the Compose file format, but is honoured by Docker Swarm,
  not by `docker compose up`. On one machine it is also mostly pointless: three replicas
  share one CPU and one memory pool, so it buys concurrency, not capacity.
- **No load balancer.** Compose's DNS returns container IPs; it does not distribute
  requests, do health-aware routing, retry a failed backend, or drain a replica being
  replaced. You would have to add nginx/Traefik as another service and configure it
  yourself.
- **No rolling updates.** Compose stops and starts containers. There is no way to bring
  up a new version, wait for it to pass health checks, shift traffic, and roll back
  automatically if it fails.
- **Published ports collide.** Three containers cannot all map to host port 8000, so any
  real scale-out needs a routing layer in front.

**For mlflow to survive a machine failure:**

- **The named volume is local.** It lives on one machine's disk. If that machine dies, the
  volume dies. Surviving failure means the state must not be on the node at all: Postgres
  as the backend store, S3/GCS as the artifact store — both of which MLflow supports, and
  both of which move the durability problem to a system designed for it.
- **SQLite is single-writer.** Even two mlflow replicas against the same volume would
  corrupt it. Horizontal scaling requires the Postgres backend first.
- **Nothing reschedules.** If the container dies, `restart: on-failure` retries on the same
  machine. If the machine is gone, nothing happens — there is no scheduler watching, no
  concept of a desired state to reconcile.
- **No secrets management.** Credentials for a real artifact store would be environment
  variables in a file committed to git.

What is actually missing is a **cluster orchestrator**: a control plane holding declared
desired state, a scheduler placing containers across nodes, a Service abstraction providing
a stable virtual IP with load balancing, readiness and liveness probes wired into routing,
Deployments doing rolling updates and rollbacks, PersistentVolumes backed by networked
storage, and Secrets. That is Kubernetes, and it is the next lab.

Compose is not a deficient Kubernetes — it is a correct tool for a different job. One
machine, one developer, a stack you bring up and tear down. The moment "survive a machine
failure" enters the requirements, the machine can no longer be the unit of deployment.

---

## Files produced in this lab

| File | Purpose |
|---|---|
| `mlflow/Dockerfile` | Tracking server + registry image, with artifact proxying and the Host-header allow-list. |
| `frontend/app.py` | Streamlit upload page; reads `INFERENCE_URL` from the environment. |
| `frontend/Dockerfile` | Streamlit image, run headless so it doesn't prompt on stdin. |
| `docker-compose.yml` | Wires the three services, the healthcheck gate and the named volume. |
| `lab/lab4.md` | This file. |

### A methodological note: verify through the right layer

Q8 and Q9 were first attempted through the browser, and produced a misleading result: the
prediction stayed at "Soup 26.8%" across an alias change *and* a container restart, which
would have meant the two model versions were identical.

They were not. Loading both versions inside the inference container and predicting on a
fixed input settled it immediately:

```
version 1  [-0.188  0.691 -0.082 -0.571 -0.9  ]
version 2  [ 0.323  0.506  1.134  0.454  0.318]
```

Completely different models. The stale reading came from Streamlit, which only re-runs its
script when something changes — a page reload with the file still in the uploader re-drew
the previous result without sending a new request.

Redoing the experiment through `docker compose exec inference python -c "requests.post(...)"`
gave the clean four-reading table in Q8. The lesson is not about Streamlit: when measuring
the behaviour of one layer, the measurement should not pass through three others that can
each cache, retry or reorder. The frontend is the thing being demonstrated to a user; it is
not the instrument for testing what the backend loaded.
