# Propagating Knowledge in LMs Through Distillation 

Paper: https://arxiv.org/pdf/2306.09306v1.pdf

Abstract:

Modern language models have the capacity to store and use immense amounts
of knowledge about real-world entities, but it remains unclear how to update
their implicit “knowledge bases.” Prior methods for updating knowledge in LMs
successfully inject facts, but LMs then fail to make inferences based on these
injected facts. In this work, we demonstrate that a context distillation-based
approach can both impart knowledge about entities and propagate that knowledge
to enable broader inferences. Our approach consists of two stages: transfer set
generation and distillation on the transfer set. We first generate a transfer set by
simply prompting a language model to generate a continuation from the entity
definition. Then, we update the model parameters such that the distribution of the
LM (the student) distribution matches the distribution of the LM conditioned on
the definition (the teacher) on the transfer set. Our experiments demonstrate that
this approach is more effective in propagating knowledge updates compared to fine-
tuning and other gradient-based knowledge-editing methods without compromising
performance in other contexts, even when injecting multiple entities at once.

<img width="1156" alt="image" src="https://github.com/shankarp8/knowledge_distillation/assets/47063867/2d111b0d-9067-48d4-86ab-8f318355d629">



