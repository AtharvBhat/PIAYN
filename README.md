# PIAYN
Perceiver is all you need ?

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
tain minimal inductive biases for long-range
vision tasks, with limited assumptions about
the input. However, it has not been tested on
text data, especially for long-range tasks. Fur-
thermore, the effects of architectural inductive
biases on long context tasks have not been ex-
plored. This is what we attempt to test in this
work
