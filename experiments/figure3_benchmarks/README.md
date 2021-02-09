## Benchmark Experiments (Figure 3)
These experiments are built on the implementation of Liu et al. (2020), see  
[https://github.com/fengliu90/DK-for-TST](https://github.com/fengliu90/DK-for-TST).

The baselines are created using the `freqopttest` library which has to be installed as


```pip install git+https://github.com/wittawatj/interpretable-test```

To run the implementations on CPU, you have to set the hardcoded parameter `is_cuda` to False

####Note: 
Since training the deep methods is rather slow, the error are estimated over 10 runs of the training, and each time
the error is estimated with 100 test sets. For the KFDA implementation, we directly estimate the error over 100 runs each 
times, each time recomputing the KFDA-witness.

### Blobs
Run the following 3 files:
`blobs_kfda_witness.py`, `Deep_kernel_Blob.py`, and `Baselines_Blob.py`.
After completion, evaluate using `evaluation_Blobs.py`.

### HIGGS
You first need to download the data, and place it in this directory.
[https://drive.google.com/open?id=1sHIIFCoHbauk6Mkb6e8a_tp1qnvuUOCc](https://drive.google.com/open?id=1sHIIFCoHbauk6Mkb6e8a_tp1qnvuUOCc)

Run the following 3 files: `higgs_kfda_witness.py`, `Deep_kernel_HIGGS.py`, and `Baselines_HIGGS.py`.
After completion, evaluate using `evaluation_Blobs.py`.
