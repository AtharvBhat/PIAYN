
## LRA Benchmark Experiments

Place the PIAYN_Pytorch folder into your scratch folder, and rename it PIAYN.

Before running the tasks, one has to create the dataset files for each tasks following the instructions in the `data` folder, and add the files in the `/datasets/{task}` folder. 

To run the LRA tasks, one would need
```
pytorch==1.7.1, transformers==3.3.1, performer-pytorch, perceiver-pytorch
```
To run a LRA experiment, run the following command in `code` folder
```
python3 run_tasks.py --model <model> --task <task> --netid <netid>
```
where `<model>` can be set to `linear, linformer-256, performer-256, perceiver` corresponding to standard self-attention and cross-attention, Linear Attention, Linformer with the projection dimension as 256, Performer with 256 random feature dimension. And `<task>` can be set to `listops, text, retrieval`. The best models and log files will be saved `LRA/logs/` folder.
