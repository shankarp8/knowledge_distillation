# Propagating Knowledge in LMs Through Distillation 

This is the official repository of the paper: 
> [Propagating Knowledge Updates to LMs Through Distillation](https://arxiv.org/pdf/2306.09306v1.pdf) <br/>
> Shankar Padmanabhan, Yasumasa Onoe, Michael J.Q. Zhang, Greg Durrett, Eunsol Choi <br/>
> NeurIPS 2023

## Getting Started 

This codebase uses Python 3.7.9. 

```
$ conda create -n knowledge_distill -y python=3.7.9
$ conda activate knowledge_distill
(knowledge_distill) $ pip install -r requirements.txt
```
## Running experiments

To run experiments, run a file from the root directory. There are two files, one for Entity Inferences and one for ECBD.

Example: 
```
(knowledge_distill) $ python experiments/gpt_ecbd.py
```
To choose the editing method, change 'ki_method' in experiments/gpt_ecbd.py or experiments/gpt_entity_inferences.py accordingly. 

Compatibility with ROME and MEND will be added soon.

## Citing the paper
```
@article{padmanabhan_2023_distill,
  title={Propagating Knowledge Updates in LMs Through Distillation},
  author={Shankar Padmanabhan and Yasumasa Onoe and Michael J.Q. Zhang and Greg Durrett and Eunsol Choi},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}
```
## Contact
Please contact shankarpadmanabhan@utexas.edu if you have any questions or concerns.


