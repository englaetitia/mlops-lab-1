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
