## Instructive Experiments (Figure 2 + Figure 6)
To reproduce Figure 2, please run the experiment files in the subdirectories, e.q., `python fig2_left.py`. This will create
a file with the results. After successfull completion, the results 
can be evaluated running `python plot.py` in the respective directory.

For the paper the results are averaged over 1000 iterations. This might take a while. Therefore we recommend to set 
the number of iterations to 100.

### Type-I errors
To estimate type-I errors, simply change the `np.pi/4` to `0` when calling the method `generate_data` 
(e.g., in line 42 of `fig2_left.py`).