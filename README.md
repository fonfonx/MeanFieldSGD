# MeanFieldSGD

Code for the paper **Quantitative Propagation of Chaos for SGD in Wide Neural Networks** by Valentin De Bortoli, Alain Durmus, Xavier Fontaine and Umut Şimşekli, in *Advances in Neural Information Processing Systems*, 2020.

See the [paper](https://papers.nips.cc/paper/2020/hash/02e74f10e0327ad868d138f2b4fdd6f0-Abstract.html) for more details.

## How to use the code

Run the `run.py` file by specifying which parameters to use (width, batch size, etc.) . It will produce a `jobs` file. Then run all the jobs of this file, for example with the `parallel` tool: `parallel < jobs`. This will create log files into a folder name `width_exp_T100`.

Once the networks are trained, the weights are saved into the `width_exp_T100` folder. Run the ipython notebook `visualize_results.ipynb` to display the histograms.

## Credits

The code is based on [the following repository](https://github.com/umutsimsekli/sgd_tail_index).
