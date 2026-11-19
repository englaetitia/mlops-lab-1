# Lab 5 - CI/CD with GitHub Actions

**Repository:** https://github.com/englaetitia/mlops-lab-1
**Author:** Laetitia Daou

## Making the code testable

- `src/food11/data.py` now has `CLASSES` (the 11 names), `category_from_filename("3_120.jpg") -> "Egg"` and `resize_image(img, size=128)`. The copy script uses `resize_image` too, so the tests check the same function the real script uses.
- `src/food11/serve.py` keeps the model in a module-level variable `model`. If `SKIP_MODEL` is set, it starts without loading anything from MLflow: `/health` still answers 200 and `/predict` answers 503.
- `pyproject.toml` has `pythonpath = ["."]` for pytest (so `from src.food11... import` works) and a `[tool.ruff.lint]` section, so ruff checks the same rules on my laptop and in CI. I had to add `known-third-party = ["mlflow"]` because my `mlflow/` folder (lab 4 Dockerfile) made ruff think `mlflow` was my own package.

## Question 1 - Why is it a bad idea for a unit test to load the real model from MLflow?

A unit test should only check my own code (functions and classes), quickly and without depending on anything outside. Loading the real model from MLflow turns it into an integration test. The test could fail even if my code is correct, for example:
- the MLflow server is not running (on my laptop, or in GitHub Actions where there is no server at all), so the connection is refused
- the `champion` alias doesn't exist or points to another version (like after `docker compose down -v` in lab 4)
- the download of the model (~45 MB) times out or the network is slow

It would also be slow, and not repeatable: the result changes every time the champion changes. That's why `serve.py` has `SKIP_MODEL` and the test uses a fake model.

## Question 2 - What does `test_predict_returns_valid_json` prove, and what does it not prove?

The fake model always gives class 0 the highest score, so getting "Bread" back has nothing to do with the picture. The test checks **my code around the model**, not the model.

**It proves:**
- `/predict` accepts an uploaded file and reads it as an image
- the preprocessing (RGB, resize to 128, normalization, shape `1x3x128x128`) runs without crashing
- my code finds the best class in what the model returns and turns index 0 into the name "Bread"
- the answer is valid JSON with the right keys, and `confidence` is between 0 and 1

**It does not prove:**
- that the real model is good or predicts the right food: "Bread" comes from the fake, not from the image
- that the model can be loaded from MLflow (`models:/food11@champion`), because loading is skipped with `SKIP_MODEL`
- that the preprocessing is the same as in training. If I changed the normalization in `serve.py` only, the test would still pass, but the real model would get worse without any error (training/serving skew)
- that the real model accepts this input or answers fast enough

Small detail: the fake returns 0.90 for Bread but the API answers about 0.20, because `serve.py` applies a softmax (it expects raw scores, not probabilities). The test only checks that the confidence is between 0 and 1, so it doesn't see this.

So it's a unit test of the API code. To check the real model you need other tests, like the "41/55 correct" check I did in lab 3.

## Validating the data

`src/food11/validate_data.py` checks a processed dataset and prints one PASS/FAIL line per check, then exits with code 1 if anything failed:
1. each split has exactly the 11 class folders
2. every image opens with PIL (fully decoded), is RGB and 128x128
3. every class has at least `--min-per-class` images in each split (default 50)
4. no image is in two splits (md5 of the file content)
5. each class's share of training is within `--tau` (default 0.05) of its share of validation

**First run, before breaking anything: it already failed.** Check 4 found 43 images that were in two splits at the same time in `food11_processed_mini`, for example:

```
FAIL  4. no image in two splits
        - duplicate in ['evaluation', 'training']: ...\training\Bread\0_106.jpg, ...\evaluation\Bread\0_131.jpg
        - duplicate in ['training', 'validation']: ...\training\Dairy product\1_185.jpg, ...\validation\Dairy product\1_123.jpg
        ...
```

I checked the original files: `data_local/food11_raw/training/0_106.jpg` and `evaluation/0_131.jpg` have the same md5. So the duplicates come from the **original Food-11 download**, not from my script. My `data.py` just copied them.

I fixed it in `data.py`: it now keeps the md5 of every image it uses, goes through the splits in order (training, evaluation, validation), and skips an image if the same content was already used. A duplicate stays in training and is removed from the test and validation splits. After rebuilding, it skipped **161 duplicates** (145 in validation, 15 in evaluation, 1 inside training), and both datasets pass:

```
PASS  1. 11 class folders in each split
PASS  2. images open, RGB, 128x128
PASS  3. at least 50 images per class
PASS  4. no image in two splits
PASS  5. class balance train vs val (tau=0.05)

RESULT: PASS
```

Then I broke it on purpose: I copied one image from `training/Bread` into `evaluation/Bread` with another name (`copy_0_0.jpg`):

```
FAIL  4. no image in two splits
        - duplicate in ['evaluation', 'training']: ...\training\Bread\0_0.jpg, ...\evaluation\Bread\copy_0_0.jpg

RESULT: FAIL
> echo $LASTEXITCODE
1
```

Then I removed the copy.

## Question 3 - Which check caught it? Why is a train/test duplicate more dangerous than a corrupted image?

**Check 4** (no image in two splits) caught it, even though the copy had a different name, because it compares the content and not the name.

A **corrupted image** makes the training script crash right away. It's annoying, but you see it immediately, you find the file and you fix it.

A **duplicate** between training and test doesn't crash anything. Everything runs normally, but the model is tested on pictures it already saw during training, so the accuracy looks better than it really is. Nobody notices, and you could choose or deploy a model because of a score that is partly fake. A loud error is easy to fix; a silent one is not.

My own data is the proof: the original Food-11 had 161 of these duplicates, 145 of them in validation (about 7% of the validation split). So the `val_accuracy` and `test_accuracy` I reported in lab 2 were measured partly on images the model had already seen. I only found it because of this check.

## Question 4 - The CI runner can't see the data. Where should `validate_data.py` run to still protect me?

The GitHub runner only has the git repo, so it only has `data.dvc`, not the images. The check has to run where the data is:

1. **When the data is created, before it is saved with dvc.** Right after `data.py` and before `dvc add data` / `dvc push`, for example in a git pre-commit hook or as a stage in a `dvc.yaml` pipeline. That way a broken dataset never gets pushed and never gets a commit.
2. **At the start of training, on the machine that trains.** `train.py` (or the training job) runs the check first and stops if it fails. The model is never trained or registered on bad data, even if someone changed the data by hand.

A third option for later: the CI job itself could get the data. My remote is DagsHub, so a job could run `dvc pull` with a DagsHub token stored as a GitHub secret, then run `validate_data.py`. It would be slower, but it would check the data on every push.

## The CI workflow

`.github/workflows/ci.yml` has three jobs:
- `test`: `uv sync --frozen`, then `ruff check src tests`, then `pytest -q`
- `smoke` (`needs: test`): builds the Docker image, starts it with `SKIP_MODEL=1` and calls `/health` up to 20 times. If it never answers, it prints the container logs and fails.
- `push` (`needs: smoke`, only on `main`): logs in to ghcr.io with `GITHUB_TOKEN` and pushes the image tagged with the commit SHA.

**Problem I had:** the first run failed after 2 seconds with *"The job was not started because your account is locked due to a billing issue."* My code wasn't the problem: a card saved on my GitHub account had failed its authorization check. I'm on GitHub Free, so I don't need a card. I removed it and contacted GitHub support to remove the lock.

## Question 5 - `smoke` has `needs: test`. What happens if a unit test fails? And without `needs`?

On my first run, `test` failed, and `smoke` was shown as **skipped**: with `needs: test`, a job only starts if the job it needs succeeded. So if a unit test fails, the image is never built or tested, and (since `push` needs `smoke`) never pushed.

Without `needs`, all jobs start at the same time on different runners. `smoke` would build and test the image even if the unit tests failed, which wastes minutes, and the results would be confusing (smoke green, test red). Worse, a `push` job without `needs` could push an image of broken code to the registry.

## Question 6 - How long did the run take? Second run with an empty commit: which step got faster, and why?

> TO FILL after GitHub Actions is unlocked: total time of run 1 and run 2, and the time of `uv sync --frozen` in each.

Every run starts on a fresh, empty machine, so nothing from the previous run is there unless it is cached. In the `test` job, `astral-sh/setup-uv` has `enable-cache: true`: at the end of the first run it saves uv's download cache (the wheels of torch, mlflow, numpy...), keyed on `uv.lock`. In the second run `uv.lock` didn't change, so the cache is restored and **`uv sync --frozen`** doesn't download everything again, it only installs from the cache. That's the step that gets faster. The `smoke` job uses a plain `docker build` with no cache, so it takes about the same time each run.

## Question 7 - Why tag the image with the commit SHA and not `latest`? A situation where `latest` makes two machines run different code.

`latest` is just a name that moves: every new push overwrites it. The commit SHA never changes, and it points to exactly one version of the code. With the SHA I always know which commit an image was built from, and I can go back to an old image just by using its tag. This answers Q9 of lab 3 (an immutable tag that another machine can pull).

**Example:** on Monday, server A does `docker pull food11-api:latest` and gets the image of commit `abc123`. On Tuesday I push a change to `serve.py` (commit `def456`), CI pushes a new `latest`. On Wednesday, server B starts and pulls `latest`: it gets `def456`. Now A and B both say they run `latest`, but they run different code. If B has a bug, I can't easily tell which code is running where, and I can't roll back to "the old `latest`" because it was overwritten. With `ghcr.io/englaetitia/mlops-lab-1:abc123`, both machines run exactly what I asked for.

## Question 8 - Why does the `push` job declare `permissions:`? What could a compromised step do with a token that can write everything?

`GITHUB_TOKEN` is created for each run, and `permissions:` sets what it is allowed to do. The `push` job only needs two things: read the code (`contents: read`) and write to the registry (`packages: write`). Writing them explicitly follows the **least privilege** idea: the token can't do more than the job needs.

The job uses actions written by other people (`docker/login-action`, `docker/build-push-action`...). If one of them were compromised (for example a bad new version), it would run with my token. With write access to everything, it could:
- push code to `main` directly, for example add a backdoor in `serve.py`
- change `.github/workflows/ci.yml` so that every future run does something bad
- create or delete branches, tags and releases, or change issues and pull requests
- publish a malicious image under my name

With only `contents: read` and `packages: write`, the worst it can do is limited to the registry, and the code itself stays safe.

## Question 9 - Pull request with a broken test: what does the page show? Could you still merge? What does it change for a team?

Branch protection rule on `main` (Settings → Branches): **Require a pull request before merging** (without required approvals, since I work alone) and **Require status checks to pass before merging** with the checks `test` and `smoke`.

On a branch `break-test` I made `category_from_filename` return the wrong class, pushed it and opened a pull request.

> TO FILL after GitHub Actions is unlocked: screenshot of the pull request page.

The pull request page shows the checks running, then `ci / test` failing (red) and `ci / smoke` skipped, with the message that required checks have not passed, and the **Merge** button is blocked. A normal contributor cannot merge. As the repo owner (admin) I still see an option to bypass the rule and merge anyway, because I didn't enable "Do not allow bypassing the above settings"; in a real team that option would be on so nobody can skip the checks.

What it changes for a team: with "everybody runs the tests on their laptop before pushing", the tests only run if each person remembers, on their own machine with their own setup, and nobody can see if they did. With required checks, the tests always run, in the same clean environment, on every pull request, and the result is visible to everyone. Broken code can't reach `main`, so `main` is always in a working state, and the review can focus on the change itself instead of "did you run the tests?".

Then I fixed the test, pushed again on the same branch, the checks turned green and I merged.

## The quality gate (`promote.py`)

`src/food11/promote.py --version <n>`:
1. reads `test_accuracy` of the run behind the candidate version and of the run behind the current `champion`
2. computes `se = sqrt(p(1-p)/n)` with `p` = champion accuracy and `n` = size of the test split (`train.py` now logs it as the param `test_images`)
3. if `acc_candidate >= acc_champion + 2*se`, it moves `champion` to the candidate; otherwise it sets `challenger` on the candidate and `champion` stays
4. in both cases it writes the decision and the numbers as tags on the candidate version (`gate_decision`, `candidate_accuracy`, `champion_accuracy`, `gate_threshold`, ...)

I trained two new runs on the clean mini dataset and registered them:

```
lr=0.00001 -> test_accuracy=0.5484  -> version 2
lr=0.0001  -> test_accuracy=0.7737  -> version 3
```

## Question 10 - What did the gate decide for each candidate? How many accuracy points is `2*se`?

```
> uv run python src/food11/promote.py --version 2
candidate v2: accuracy 0.5484
champion  v1: accuracy 0.7783  (n=1096)
se = 0.0125   2*se = 0.0251   threshold = 0.8034
decision: challenger (v1 stays champion)

> uv run python src/food11/promote.py --version 3
candidate v3: accuracy 0.7737
champion  v1: accuracy 0.7783  (n=1096)
se = 0.0125   2*se = 0.0251   threshold = 0.8034
decision: challenger (v1 stays champion)
```

- **v2** (lr 0.00001): 54.8%, far below the threshold of 80.3%. It's just a model that learned too slowly in 5 epochs. **Challenger**, v1 stays champion.
- **v3** (lr 0.0001, my best settings): 77.4%, almost the same as the champion (77.8%) but below the threshold. **Challenger** too.

With my test split of **n = 1096** images, `2*se = 0.0251`, so about **2.5 accuracy points**. A new model has to be at least 2.5 points better than the champion to replace it. A difference smaller than that could just be luck in which test images are easy or hard, so the gate doesn't trust it. v3 shows this well: it's 0.5 points lower than v1, which is noise, not a real difference.

Note: `challenger` is an alias, so it can only point to one version. After the second run it moved from v2 to v3, but the decision tags stay on both versions.

One limit of this comparison: the champion's 0.7783 was measured on the **old** evaluation split, which still had duplicates of training images (see Q3), while v2 and v3 were tested on the **clean** split. So the champion's score is probably a bit too high, and the comparison is not completely fair. To be correct, the champion should be re-evaluated on the same clean test set before comparing.

## Question 11 - `promote.py` can't run in GitHub's runners while MLflow is on my laptop. Two ways to run it automatically after training?

1. **Put MLflow somewhere the runner can reach.** DagsHub gives a hosted MLflow server for each repo. If I log my runs there instead of to my laptop, a GitHub workflow can run `promote.py` with `MLFLOW_TRACKING_URI` pointing to DagsHub and the token stored as a GitHub secret. It can be started by hand (`workflow_dispatch`) or on a schedule.
2. **Run it where MLflow is.** Either call `promote.py` at the end of the training script/pipeline on my machine (for example as the last stage of a `dvc.yaml` pipeline, after training and registering), or install a **self-hosted GitHub runner** on my laptop (or a server next to MLflow). The workflow then runs on that machine, which can reach `http://127.0.0.1:5000`.

## Question 12 - The pipeline from `git push` to a running container

```
 [HUMAN]  edit code ──► git push (branch) ──► open a pull request
                                │
                                ▼
 [AUTO]   GitHub Actions: ci.yml
            job test : uv sync --frozen ─► ruff ─► pytest
                │ (needs)
                ▼
            job smoke: docker build ─► docker run SKIP_MODEL=1 ─► curl /health
                │
                ▼
 [AUTO]   branch protection: merge is blocked until test + smoke are green
                │
 [HUMAN]  merge the pull request into main
                │
                ▼
 [AUTO]   ci.yml runs again on main
            test ─► smoke ─► job push: build + push
                                ghcr.io/englaetitia/mlops-lab-1:<commit SHA>
                                ▲
                                └── CODE VERSION decided here (= commit SHA)

 ── separately, the model ───────────────────────────────────────────────

 [HUMAN]  data.py ─► validate_data.py ─► dvc add / dvc push      (data version = data.dvc)
 [HUMAN]  train.py (on my GPU) ─► run logged in MLflow
 [HUMAN]  register the run as a new version of food11
 [AUTO*]  promote.py: quality gate
            acc_candidate >= acc_champion + 2*se ?
              yes ─► alias champion moves to the candidate
              no  ─► alias challenger
                ▲
                └── MODEL VERSION decided here (= alias @champion)

 ── running it ──────────────────────────────────────────────────────────

 [HUMAN]  docker pull ghcr.io/englaetitia/mlops-lab-1:<SHA>
          docker compose up   (or restart inference)
                │
                ▼
          container starts ─► loads models:/food11@champion from MLflow ─► serves /predict
```

- **Automatic:** lint, tests, image build, container smoke test, blocking the merge if checks fail, pushing the image with the SHA tag, and the promotion decision itself.
- **Still human:** writing code, opening and merging the pull request, preparing the data and running `validate_data.py`, training (it needs my GPU), registering the run, *starting* `promote.py` (marked AUTO* because the decision is automatic but I still launch it by hand, since MLflow is on my laptop, see Q11), and deploying (pulling the image and restarting the container).
- **Code version** is decided by the **commit SHA** in the image tag. **Model version** is decided by the **`champion` alias** in the registry. They are independent: the same image can serve a new model after a restart, and a new image can serve the same model.
