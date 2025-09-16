# CliqueGen + ZoningILP for Micro-Transit Zoning
This repository contains the implementation of the CliqueGen + ZoningILP algorithm for micro-transit zone optimization, as presented in the paper *Optimal Micro-Transit Zoning via Clique Generation and Integer Programming*. The paper has been accepted for IEEE ITSC 2025, and a preprint is available [here](https://www.arxiv.org/abs/2509.11445).

## Contributors
Hins Hu (zh223@cornell.edu), Rhea Goswami (rkg62@cornell.edu)

## Set up the environment
To help users run our code seamlessly, we created a config file called `environment.yml` for users to rebuild the same conda environment we used to implement the algorithm. To begin, you will need to install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) on your local machine. After that, you can modify the prefix (i.e., last line) in `environment.yml` to your customized path and create a virtual conda environment by running the following command.
```    
conda env create -f environment.yml -n YourName
```
This will create a conda environment named `YourName` with all the dependencies installed.

## How to run the code
Run the following command to activate the conda environment you just created.
```
conda activate YourName
```

Then, adjust the parameters in `main.py` as needed. They include the following:
- `NUM_ZONES`: Number of zones to create.
- `MAX_DIAMETER`: Maximum diameter of each zone.
- `ALGO`: Algorithm to use, either "clique_generation" or "baseline".
- `NUM_NODE`: Number of nodes in the synthetic network.
- `SEED`: Random seed for reproducibility.
- `ONE_WAY_PROB`: Probability of one-way edges in the synthetic network.
- `EDGE_RATIO`: A hyper-parameter to control the density of the synthetic network. Refer to the function `generate_synthetic_map` for more details.
- `CENTER`: Center coordinates of the synthetic demand clusters.
- `RADIUS`: Radii of the synthetic demand clusters.

Be caveat that your numbers may vary from ours shown in the paper due to the randomness in generating the synthetic demand pattern and network. However, the overall conclusion and insights should remain consistent.

Last, run the following command to execute the code.
```
python main.py
```

## Expected Output
The solution will be shown directly in the terminal, and the visualization of demand pattern and optimal zones will be saved in `output/`.
