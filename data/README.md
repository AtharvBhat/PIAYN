Steps to setup data.


* Step 0: cd here into the data directory
* Step 1: Clone Nystromformer repo here: https://github.com/mlpen/Nystromformer
* Step 2: Clone long-range-arena repo here : https://github.com/google-research/long-range-arena
* Step 3: Download long range arena dataset archive from the github link
* Step 4: IMPORTANT : Pathfinder files are HUGE .. they are more than a million files and greene has a cap on number of files. We will only be 
extracting listops and retrieval dataset. since they are the only langauge tasks. IMDB classification is built into tf_datasets. Hence you will also 
need to install tensorflow, tensorflow_datasets and tensorflow_text. 

To avoid extracting pathfinder tasks from the long range arena use the following commands
``` 
tar -xzvf lra_release.gz lra_release/lra_release/listops-1000
tar -xzvf lra_release.gz lra_release/lra_release/tsv_data
```

* Step 5: Once you have extracted lra_release.gz and cloned long-range-arena, move both directories into Nystromformer/LRA/datasets
* Step 6: Execute text.py, retrieval.py, listops.py

After following these steps, you should have gotten pickled dev, train, test splits of all three (IMDB classification, listops and retrieval) tasks.
move all files back into the PIAYN/data folder and put them in their own directories (listops, retrieval, text)
