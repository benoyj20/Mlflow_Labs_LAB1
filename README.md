# MLflow Lab 1

This lab is about using MLflow to track experiments, save and register models, and serve a model as a REST API.

The original lab had four parts (`starter.ipynb`, `linear_regression.ipynb`, `serving.ipynb`, `serving.py`). I got all of them running, and the part I changed is `linear_regression.ipynb`. The original version trained one ElasticNet model with fixed settings, and to serve it you had to copy a run ID from the MLflow UI into a command. I changed it into a small pipeline:

1. run a hyperparameter sweep,
2. pick the best model automatically,
3. register it in the Model Registry as the "champion" (only if it's better than the current one),
4. serve whatever the champion is,
5. check the served model with a test script.

More detail on this is in the [What I changed](#what-i-changed) section near the end.

## Files

```
Mlflow_Labs_LAB1/
├── README.md
├── requirements.txt
├── .gitignore
├── linear_regression.ipynb   <- my main change (sweep + registry + champion alias)
├── wine_data.py              <- new: loads and splits the wine dataset
├── test_endpoint.py          <- new: tests the served model
├── starter.ipynb             <- original lab (small fixes)
├── serving.ipynb             <- original lab (small fixes)
└── serving.py                <- original lab (small fix for Windows)
```

When you run things, these also get created (they're in `.gitignore` so they aren't in the repo):

- `mlflow_lab1_env/` - the virtual environment
- `mlflow.db` - the database MLflow stores runs and registered models in
- `mlruns/` - saved models and plots

## What you need to install

- **Python 3.11.** I used 3.11.8. I pinned the package versions in `requirements.txt` and only tested them on 3.11, so I'd stick with that.
  - Windows: download it from https://www.python.org/downloads/windows/ and tick "Add python.exe to PATH" in the installer.
  - Mac/Linux: python.org, `brew install python@3.11`, or your package manager.
- **Git**, to clone the repo (or just download the ZIP).
- **Internet**, because pip downloads the packages and the notebooks download the wine dataset from GitHub.

To check Python is installed:

```powershell
py -3.11 --version        # Windows
python3.11 --version      # Mac/Linux
```

## Setting up

Run all of these from inside the `Mlflow_Labs_LAB1` folder (the one you get after cloning). I did everything on Windows in PowerShell, so the Mac/Linux commands are the standard equivalents.

**1. Clone the repo and go into the folder**

```bash
git clone https://github.com/benoyj20/Mlflow_Labs_LAB1.git
cd Mlflow_Labs_LAB1
```

**2. Create the virtual environment**

```powershell
py -3.11 -m venv mlflow_lab1_env          # Windows
python3.11 -m venv mlflow_lab1_env        # Mac/Linux
```

**3. Activate it**

```powershell
.\mlflow_lab1_env\Scripts\Activate.ps1    # Windows
source mlflow_lab1_env/bin/activate       # Mac/Linux
```

You should see `(mlflow_lab1_env)` at the start of your prompt. You have to do this in every new terminal you open for this lab. I forgot once and the model server crashed (see [Problems I ran into](#problems-i-ran-into)).

If PowerShell says "running scripts is disabled on this system", run this once and then try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**4. Install the packages**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs MLflow 3.17.2, scikit-learn 1.9.1, pandas, NumPy, matplotlib, XGBoost and JupyterLab. The versions matter here, because MLflow 3 works differently from MLflow 2 in a few ways that the code depends on.

**5. Add the environment as a Jupyter kernel**

```bash
python -m ipykernel install --user --name=mlflow_lab1_env
```

You can check it worked with `jupyter kernelspec list`, which should show `mlflow_lab1_env`.

## Running my part (sweep, registry and serving)

You need three terminals for this, all in the `Mlflow_Labs_LAB1` folder with the environment activated.

### Step 1: Run the notebook (terminal 1)

Start JupyterLab:

```bash
jupyter-lab
```

Open `linear_regression.ipynb`, make sure the kernel (top right) is `mlflow_lab1_env`, and do **Run > Run All Cells**. If it's not on the right kernel, change it under **Kernel > Change Kernel**.

If you don't want to open JupyterLab, this runs the whole notebook from the terminal instead:

```bash
jupyter nbconvert --to notebook --execute linear_regression.ipynb --inplace
```

It takes about 2 minutes for me, mostly because it saves 12 models.

**What you should see:**

- One line for each of the 12 hyperparameter combinations, like:
  ```
  alpha=0.01, l1_ratio=0.1     RMSE=0.6742  MAE=0.5166  R2=0.3560
  ...
  alpha=1.0, l1_ratio=0.9      RMSE=0.8329  MAE=0.6718  R2=0.0171
  ```
- A table of the 5 best runs sorted by RMSE. The best one is `alpha=0.01, l1_ratio=0.1` with RMSE 0.674.
- The first time you run it:
  ```
  Promoted version 1 (RMSE 0.6742) to champion. Previous champion: none
  ```
- If you run it again, the new best model is the same, so it isn't better than the champion and doesn't replace it:
  ```
  Version 2 (RMSE 0.6742) did not beat champion version 1 (RMSE 0.6742); labelled it challenger
  ```
- The last cell loads the champion from the registry and prints 5 predictions.

### Step 2: Serve the champion model (terminal 2)

```bash
mlflow models serve -m "models:/ElasticnetWineModel@champion" --env-manager=local -h 127.0.0.1 -p 5001
```

After a few seconds it should say `Uvicorn running on http://127.0.0.1:5001`. Leave it running. You don't need a run ID here because `@champion` always points to whichever version is the champion.

### Step 3: Test it (terminal 3)

```bash
python test_endpoint.py
```

This is what I got (your version number and run ID will probably be different):

```
Champion: ElasticnetWineModel version 1 (run 0432c85b8b87430891c61d5f6cf10938)
Rows scored by endpoint: 400
First 5 served predictions: [5.925, 5.088, 5.764, 6.409, 5.443]
RMSE of served predictions on test set: 0.6742
PASS: served predictions match the champion model loaded from the registry
```

If it prints `PASS`, it worked. It exits with code 0 for PASS and 1 for FAIL. You can stop the server with Ctrl+C in terminal 2 after this.

### Step 4: Look at it in the MLflow UI

```bash
mlflow ui
```

Then go to http://127.0.0.1:5000. Things to look at:

- **Experiments > wine-elasticnet-sweep:** there's one parent run called `elasticnet-sweep`. If you expand it you'll see the 12 child runs. You can select them and click **Compare** to see how RMSE changes with `alpha` and `l1_ratio`.
- **A child run > Artifacts:** the residual plot (`residuals.png`) and the saved model, with its signature and input example.
- **The parent run:** the dataset that was used and a `best_run_id` tag.
- **Models > ElasticnetWineModel:** all the versions, which one is `@champion` / `@challenger`, and the RMSE tag on each.

## Running the original lab parts

These are the lab's original files. I only made small fixes so they'd run on MLflow 3 and Windows. Same as before, run them from `Mlflow_Labs_LAB1` with the environment activated.

**starter.ipynb.** Open it in JupyterLab with the `mlflow_lab1_env` kernel and run the cells in order, but **skip the `!mlflow ui --port=5001` cells**. They start a server that never finishes, so the notebook just hangs. Run `mlflow ui` in a separate terminal instead. If it worked, the "Store ML models" cell prints predictions and a `Run ID`, and the last cell loads that model and prints predictions again.

**serving.ipynb.**
1. Run the cells up to and including Step 4 ("Train and Log the Model"). It prints `Run ID: <run_id>`.
2. In another terminal, run the server with that run ID (make sure nothing else is using port 5001 first):
   ```bash
   mlflow models serve --env-manager=local -m runs:/<run_id>/model -h 127.0.0.1 -p 5001
   ```
   Skip the `!mlflow models serve` cell in the notebook, for the same reason as the `!mlflow ui` cells.
3. Go back to the notebook and run the Step 7 cells. The last one should print something like `{'predictions': [139.54..., 179.51..., ...]}`.

**serving.py.**

```bash
python serving.py
```

It logs an XGBoost model 6 different ways to show how MLflow records the pip requirements for a model. If it finishes without a traceback, all of its `assert` checks passed (MLflow might print some INFO or warning lines, that's fine). In the MLflow UI, the newest run under **Default** has 6 model artifacts, each with its own `requirements.txt`.

## What I changed

### My main change: linear_regression.ipynb

The original notebook trained a single ElasticNet with `alpha=0.5` and `l1_ratio=0.5`, logged it, and that was it. To serve it you had to find the run ID yourself and paste it into a path like `mlruns/0/<run_id>/artifacts/model`. I wanted the notebook to actually decide which model is best and make that the one that gets served, so I changed it to do this:

- **Hyperparameter sweep with nested runs.** It tries every combination of `alpha` in [0.01, 0.1, 0.5, 1.0] and `l1_ratio` in [0.1, 0.5, 0.9]. Each one is a child run under one parent run, so the whole sweep is grouped together in the UI instead of being 12 separate runs.
- **More logging.** Each run logs its params and metrics like before, plus a residual plot (`mlflow.log_figure`) and the model with a signature and an input example. The parent run logs the dataset with `mlflow.log_input`.
- **Picking the best model in code.** I use `mlflow.search_runs` to get the child runs of this sweep sorted by RMSE, and take the first one. The winner is saved on the parent run as a tag.
- **Registering with champion/challenger aliases.** The best model is registered as a new version of `ElasticnetWineModel`. Then it checks the current champion's RMSE. If the new one is lower, it gets the `champion` alias. If not, it gets `challenger` and the champion stays the same. This way rerunning the notebook can never make the served model worse.
- **Serving by alias.** Because of the alias, the serve command is always `models:/ElasticnetWineModel@champion` and never changes. When a better model gets promoted, restarting the server picks it up.
- **test_endpoint.py (new).** It sends the test set to the running server and compares the results with the champion model loaded directly from the registry. That proves the server is really serving the champion, instead of me just reading the JSON output.
- **wine_data.py (new).** I moved the data loading and train/test split here (with a fixed `random_state=40`), so the notebook and `test_endpoint.py` use exactly the same 400 test rows.

The best model from the sweep (`alpha=0.01, l1_ratio=0.1`) gets a test RMSE of 0.674. The original lab's settings got 0.793.

### Fixes to get the original lab working

The lab was written for an older MLflow on Mac/Linux, so a few things broke for me on Windows with MLflow 3.17. These are just fixes, not my main change:

- **requirements.txt:** pinned all the versions and added `jupyterlab` (it wasn't in the list, so `jupyter-lab` didn't exist), plus `pandas`, `matplotlib` and `requests`, which the code uses directly.
- **starter.ipynb, cell 8:** MLflow 3.17 saves scikit-learn models with a library called skops, which refuses to save a RandomForest unless you mark its tree type as trusted. I added `skops_trusted_types=["sklearn.tree._tree.Tree"]` to `log_model`.
- **starter.ipynb (last cell) and serving.ipynb (serve cell):** these had run IDs from whoever made the lab hard-coded in them. Also, MLflow 3 doesn't save models at `mlruns/0/<run_id>/artifacts/model` anymore. I changed them to `runs:/<run_id>/model` using the run from the current session.
- **serving.ipynb, training cell:** now prints the run ID so you know what to serve.
- **serving.py:** crashed on Windows with a `PermissionError`. `tempfile.NamedTemporaryFile` keeps the file locked on Windows, so MLflow couldn't open it. I changed it to `delete=False` and delete the file with `os.remove` afterwards.
- **All notebooks:** set the kernel to `mlflow_lab1_env` so you don't have to pick it.

## Problems I ran into

- **`No module named 'mlflow'` when starting the model server.** This happened when I hadn't activated the environment in that terminal. MLflow starts a program called `uvicorn`, and it was finding the copy from my other Python install. Activating the environment fixed it.
- **`Registered Model with name=ElasticnetWineModel not found`.** You need to run `linear_regression.ipynb` before serving or testing. Also run everything from the `Mlflow_Labs_LAB1` folder, because MLflow looks for `mlflow.db` in the folder you're in.
- **`ConnectionError ... 127.0.0.1:5001`.** The model server isn't running (or hasn't finished starting). Start it and wait for `Uvicorn running on ...`.
- **Port 5001 already in use.** Something else is on that port, like the `serving.ipynb` server or a `!mlflow ui --port=5001` cell. Stop it with Ctrl+C, or use a different port: add `-p 5002` to the serve command and run `python test_endpoint.py --url http://127.0.0.1:5002/invocations`.
- **A notebook cell runs forever.** That's one of the `!mlflow ui` or `!mlflow models serve` cells. Use **Kernel > Interrupt** and run the command in a terminal instead.
- **Wrong kernel in the notebook.** Run setup step 5 again, restart JupyterLab, and choose `mlflow_lab1_env` under **Kernel > Change Kernel**.
- **Can't download `winequality-red.csv`.** It comes from GitHub, so check your internet connection.
