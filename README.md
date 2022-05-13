# PIAYN
## Perceiver is all you need ?


Transformer-based architectures (Vaswani
et al., 2017) have achieved state-of-the-art per-
formance on several modern natural language
processing(NLP) tasks. However, due to the
quadratic space and time complexity of the at-
tention mechanism, their use for large input se-
quences remains limited. In recent years, many
architectures have proposed approximations of
the vanilla attention mechanism which scale
linearly with respect to the input size. How-
ever, as shown in (Tay et al., 2020a), these X-
formers also introduce inductive biases, which
prevent them from performing well on certain
long-range NLP tasks, thus raising questions
on generalizability. The Perceiver (Jaegle et al.,
2021) formulates a transformer-based frame-
work which has been empirically shown to con-
tain minimal inductive biases for long-range vi-
sion tasks, with limited assumptions about the
input. However, it has not been tested on text
data, especially for long-range tasks. We con-
ducted the experiments on Long Range Arena
proposed by (Tay et al., 2020a) and observed
that Perceiver performs well for long context
tasks.

## Data Generation and Train/Test/Val Splits

In this project, we focus on the text-based Long-Range Arena tasks, namely Text Classification, Long ListOps and Byte-Level Document Retrieval. To generate the datasets for these tasks, go to `/data` and follow the instructions in the README file to generate the `.pickle` files. 

## Baseline and Model Execution

To reproduce the results in our paper, go to `/PIAYN_Pytorch/LRA` and follow the instructions in the README file to run the experiments for each of the models and tasks. To run on Greene, rename the `PIAYN_Pytorch` folder to `PIAYN` and place it in your `/scratch` folder. 


