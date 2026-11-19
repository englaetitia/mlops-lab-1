# Lab 4 - Orchestrating the app with Docker Compose

**Repository:** https://github.com/englaetitia/mlops-lab-1
**Author:** Laetitia Daou

## What I changed compared to the lab sheet

### 1. `--artifacts-destination` + `--serve-artifacts` instead of `--default-artifact-root`

The lab's `mlflow/Dockerfile` uses `--default-artifact-root /mlflow-data/mlruns`. With that, MLflow saves the artifact location as a **path inside the mlflow container**. The inference container doesn't have `/mlflow-data`, so `mlflow.pyfunc.load_model` would look for a folder that doesn't exist. It's the same problem as in lab 3 (there it was a Windows path). An artifact location that is a file path only works for clients that share the same disk.

With `--serve-artifacts`, the location is saved as `mlflow-artifacts:/<experiment_id>` and every client downloads the files over HTTP from the tracking server. Only the mlflow container needs the volume.

### 2. `--allowed-hosts` must include the service name

MLflow checks the `Host:` header to block DNS rebinding attacks. By default it only accepts localhost and private IP ranges. The inference container calls `http://mlflow:5000`, so it sends `Host: mlflow:5000`, which isn't in the list, and every request would get **403 Invalid Host header**. I had the same problem in lab 3 with `host.docker.internal`.

```
--allowed-hosts "localhost,localhost:*,127.0.0.1,127.0.0.1:*,mlflow,mlflow:*"
```

This option replaces the default list, so localhost must be written again, otherwise the UI on `127.0.0.1:5000` stops working.

### 3. Alias instead of `Staging`

The lab asks to move the model to `Staging`. Stages are deprecated in recent MLflow (the lab 3 sheet says it too), so I kept using the `champion` alias from lab 3. Everything in the questions works the same way, only the name is different.

### 4. Reusing the lab 3 image

In `docker-compose.yaml`, the inference service has both `build: .` and `image: food11-api:latest`, so Compose uses the image I already built in lab 3. I needed this because Compose builds services **in parallel**. Building `frontend` and `inference` at the same time split my internet between them. The speed dropped to about 50 kB/s and `uv` stopped with:

```
error: Failed to download `numpy==2.5.3`
  cause: Failed to download distribution due to network timeout (current value: 30s)
```

The Dockerfile wasn't broken, the two builds were just slowing each other down. Building one service at a time fixed it, and I added `UV_HTTP_TIMEOUT=300` in the builder stage so a slow connection doesn't make the build fail.

(My compose file is called `docker-compose.yaml` instead of `docker-compose.yml`. Docker Compose accepts both names.)

---

## Question 1 - What happens to `/mlflow-data` if no volume is mounted?

Everything written there goes into the **container's writable layer**, and it's deleted with the container. I tested it by running the image alone, without `-v`:

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

`will-i-survive` is gone. The new container starts from the image, where `/mlflow-data` is empty. The SQLite file written by the first container was never part of the image and was deleted with `--rm`.

I also saw this by accident later: an image I copied into the inference container with `docker compose cp` disappeared after `docker compose down`, because `down` removes the containers. Anything that's not on a volume is temporary.

## Question 2 - Why a named volume instead of a bind mount?

A bind mount **would work**: the data would stay, and I could even open `mlflow.db` from my editor. But the named volume is better here:

1. **Docker manages it.** `docker compose up` creates it, it survives `down`, and only `down -v` deletes it. Its life is tied to the stack, not to a folder in my repo.
2. **No host path in the compose file.** A bind mount needs a path that exists on every machine. `mlflow-data:` works anywhere, `C:\Users\User\OneDrive\...` doesn't.
3. **No permission or file system problems.** On Windows, a bind mount goes through Docker Desktop's file sharing, where SQLite's file locking doesn't work well. A named volume is a normal Linux file system. Inside the container, `df -h` shows `/dev/sdd` mounted on `/mlflow-data`, separate from the container's own file system.
4. **It can't end up in git or in the build context.** A `./mlflow-data/` folder in the repo would need to be added to `.gitignore` and `.dockerignore`, and forgetting one would mean committing a database.

A bind mount is good for **code I'm editing**, where I want to see and change the files from my laptop. It's not good for **data the application owns**.

## Question 3 - Why does `http://mlflow:5000` resolve now?

Compose creates a private network for the stack, with its own **DNS server**. Each service gets a DNS name equal to its service name, pointing to its container's IP. So `mlflow` is a real hostname inside the stack, and `MLFLOW_TRACKING_URI=http://mlflow:5000` just works:

```
INFO:food11.serve:tracking uri: http://mlflow:5000
INFO:food11.serve:loading model: models:/food11@champion
INFO:food11.serve:model loaded
```

In lab 3 there was no such network. My container was alone on the default bridge and didn't know about other containers, so to reach MLflow on my laptop I had to use `host.docker.internal`.

The DNS only knows **running** containers. When I stopped mlflow on purpose, I got:

```
NameResolutionError: HTTPConnection(host='mlflow', port=5000):
Failed to resolve 'mlflow' ([Errno -5] No address associated with hostname)
```

Not "connection refused": the name doesn't exist anymore. A stopped service just disappears from the DNS.

## Question 4 - Why read `INFERENCE_URL` from an environment variable?

Because the image shouldn't depend on where it runs. Writing `http://inference:8000` in the code only works inside this compose stack:
- running the frontend image alone against an API on my laptop would need a code change and a rebuild
- renaming the service in `docker-compose.yaml` would break the frontend
- in Kubernetes the address could be something like `http://inference.default.svc.cluster.local:8000`, so I'd need another image

With an environment variable: **one image, many configurations**. The default (`http://127.0.0.1:8000`) works for local development. It's the same idea as `MLFLOW_TRACKING_URI` and `MODEL_URI` in `serve.py`: what changes between environments is config, not code.

## Question 5 - Why doesn't `inference` publish a port? How does the frontend reach it?

Publishing a port maps a container port to a **port on the host**. You only need it when something outside the Compose network has to connect. Two services need that: `mlflow` (I open its UI) and `frontend` (I open the page). Nobody outside calls the inference API directly: only the frontend does, and the frontend is **inside** the network.

The frontend reaches it at `http://inference:8000` with the same internal DNS as in Q3. The page shows `Model: models:/food11@champion`, which it can only know by calling `GET http://inference:8000/health`. So the internal connection works without any published port.

It's also more secure. A published port can be reached by anything that can reach my laptop. The inference API has no authentication, so publishing it would let anyone on the network send files and run predictions. Keeping it internal means only the two ports a human needs are open.

## Question 6 - `depends_on` waits for the start, not for readiness

With the lab's `depends_on: [mlflow]`, Compose only makes sure the mlflow container **starts** first. The MLflow server inside needs about 15 seconds (database migrations, workers) before it accepts connections. `serve.py` loads the model at startup, so it would call a server that isn't ready yet, and the container would fail.

I showed it on purpose by stopping mlflow and recreating inference:

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

My fix: a **healthcheck** on mlflow, and inference waits until mlflow is healthy:

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

Now Compose waits for MLflow to really answer before starting inference. You can see it at startup: `✔ Container m... Healthy 16.9s` before the other two start. I also added `restart: on-failure` as a backup for cases the healthcheck doesn't cover, for example when the alias doesn't exist yet.

`/health` works well for this because MLflow doesn't apply the Host header check on it.

## Question 7 - `docker compose ps`

```
NAME                      SERVICE     STATUS             PORTS
mlops-lab-1-mlflow-1      mlflow      Up (healthy)       0.0.0.0:5000->5000/tcp
mlops-lab-1-inference-1   inference   Up                 8000/tcp
mlops-lab-1-frontend-1    frontend    Up                 0.0.0.0:8501->8501/tcp
```

That's what the compose file says:
- `0.0.0.0:5000->5000/tcp` means **published**: a host port is mapped to the container port.
- `8000/tcp` means **only exposed**: the container listens on 8000 and other containers can reach it, but no host port is mapped.

`EXPOSE 8000` in the Dockerfile is just documentation of which port the app listens on. It doesn't open anything.

## Question 8 - After promoting a new version, does the prediction come from the new one?

**No, not until a restart.** I tested it through the API with one fixed Dessert image, changing one thing at a time:

| # | `champion` in the registry | Version in the container | Prediction |
|---|---|---|---|
| 1 | v1 | v1 | Dessert **0.5768** |
| 2 | v2 | v2 (after restart) | Dessert **0.1597** |
| 3 | **v1** | v2 (no restart) | Dessert **0.1597** |
| 4 | v1 | v1 (after restart) | Dessert **0.5768** |

Line 3 is the answer: the registry says version 1, but the service still answers with version 2, exactly the same number. `mlflow.pyfunc.load_model` reads the alias **once**, at startup, and keeps the model in memory. After that it never checks the registry again.

The command to load the new version:

```bash
docker compose restart inference
```

(I trained version 2 badly on purpose, with `lr=0.01`: 20.4% test accuracy against 77.8% for version 1, so the difference is easy to see. It still got this image right, but with 16% confidence instead of 58%. One correct answer doesn't mean the model is good.)

## Question 9 - Why does `restart` work without a rebuild?

Because the model isn't in the image. A restart runs the container's entrypoint again, so `serve.py`'s startup code runs again, reads `models:/food11@champion` again, and downloads the version it points to now.

**In the image:** the OS, Python, the venv, `serve.py`, and the **text** `models:/food11@champion`.

**Fetched when the container starts:** which version the alias points to, and its files.

So changing the model = a change in the registry + a restart. No `docker build`, no push, no code change. The image I built in lab 3, before version 2 even existed, serves version 2 without any change.

The downside, like in Q6: the container can't start without the tracking server. If MLflow is down, inference is down too.

## Question 10 - Is the model still there after `down` / `up`? And after `down -v`?

**After `docker compose down` and `up`: everything is still there.** After two full cycles:

```
> docker compose down
> docker compose up -d --no-build
> ... check ...
champion is version 1 | experiments: ['food11', 'Default']
```

`down` stops and removes the containers and the network, but **not** the named volume. So `mlflow.db` and `mlartifacts/` are still there, and the new mlflow container uses the same volume.

**After `docker compose down -v`: everything is gone.** The output says it:

```
> docker compose down -v
 ✔ Volume mlops... Removed
```

and on the next training run:

```
INFO mlflow.tracking.fluent: Experiment with name 'food11' does not exist. Creating a new experiment.
```

The experiment, both model versions, the `champion` alias and all the artifacts were deleted together. The inference service had no model to load anymore. I had to train and register again (about 40 s here, because the dataset is small and still on my laptop. With a real model it could take hours, or be impossible if the data were lost too).

So `-v` is the difference between "stop the stack" and "delete all its data". You have to be careful with it.

## Question 11 - What would be needed for 3 inference replicas, or for mlflow to survive a machine failure?

Docker Compose runs everything **on one machine**. That's the main limit.

**For 3 inference replicas behind a load balancer:**
- `deploy: replicas: 3` exists in the compose format, but it's meant for Docker Swarm. On one machine, 3 replicas share the same CPU and memory, so you get more parallel requests, not more power.
- **No load balancer.** Compose's DNS gives container IPs, but it doesn't spread requests, skip unhealthy replicas, or retry. I'd have to add nginx or Traefik and configure it myself.
- **No rolling updates.** Compose just stops and starts containers. It can't start the new version, wait until it's healthy, move the traffic, and roll back if something fails.
- **Port conflicts.** Three containers can't all use host port 8000, so you need a routing layer in front.

**For mlflow to survive a machine failure:**
- **The named volume is local**, on one machine's disk. If the machine dies, the volume dies. The data has to live somewhere else: Postgres for the backend store and S3/GCS for the artifacts. MLflow supports both.
- **SQLite allows only one writer.** Two mlflow replicas on the same file would corrupt it. You need Postgres first.
- **Nothing restarts it elsewhere.** `restart: on-failure` only retries on the same machine. If the machine is gone, nothing happens.
- **No secrets management.** Credentials for a real storage would end up as environment variables in a file in git.

What's missing is a **cluster orchestrator**: something that keeps the desired state, places containers on several machines, gives a stable address with load balancing (a Service), uses readiness/liveness probes, does rolling updates and rollbacks (Deployments), and has network storage (PersistentVolumes) and Secrets. That's **Kubernetes**, the next lab.

Compose is still the right tool for what it's made for: one machine, one developer, a stack you start and stop. When "survive a machine failure" becomes a need, one machine is no longer enough.

---

## Files I made in this lab

- `mlflow/Dockerfile`: the tracking server and registry image, with artifact serving and the allowed hosts.
- `frontend/app.py`: the Streamlit upload page. It reads `INFERENCE_URL` from the environment.
- `frontend/Dockerfile`: the Streamlit image, run in headless mode so it doesn't wait for input.
- `docker-compose.yaml`: connects the three services, with the healthcheck and the named volume.
- `lab/lab4.md`: this report.

### Note: test through the right layer

I first tried Q8 and Q9 in the browser, and the result was wrong: the prediction stayed "Soup 26.8%" after changing the alias **and** restarting, which would mean the two versions were the same model.

They weren't. Loading both versions inside the inference container and predicting on the same input showed it right away:

```
version 1  [-0.188  0.691 -0.082 -0.571 -0.9  ]
version 2  [ 0.323  0.506  1.134  0.454  0.318]
```

The old result came from Streamlit: it only re-runs its script when something changes, so reloading the page with the file still uploaded showed the old result without sending a new request.

I redid the test by calling the API directly with `docker compose exec inference python -c "requests.post(...)"`, which gave the clean table in Q8. When you test one part of the system, the test shouldn't go through other parts that can cache or hide the result. The frontend is for the user, it's not the right tool to check what the backend loaded.
