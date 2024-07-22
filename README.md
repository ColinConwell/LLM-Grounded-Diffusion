# LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models
***Transactions on Machine Learning Research (TMLR)***, with **Featured Certification**

[Long Lian](https://tonylian.com/), [Boyi Li](https://sites.google.com/site/boyilics/home), [Adam Yala](https://www.adamyala.org/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/) at UC Berkeley/UCSF

Useful Links: [Paper](https://arxiv.org/abs/2305.13655) | [Project Page](https://llm-grounded-diffusion.github.io/) | [Original GitHub Repo](https://github.com/TonyLianLong/LLM-groundedDiffusion) | [Citation](#citation)

(This fork was developed for updated imports, easier testing and generation.)

**The PipeLine**: Text Prompt -> LLM as a Request Parser -> Intermediate Representation (such as an image layout) -> Stable Diffusion -> Image.

![Main Image](https://llm-grounded-diffusion.github.io/main_figure.jpg)
![Visualizations: Enhanced Prompt Understanding](https://llm-grounded-diffusion.github.io/visualizations.jpg)

## Acknowledgements
This repo uses code from [LLM-groundedDiffusion](https://github.com/TonyLianLong/LLM-groundedDiffusion) (the original repo), [diffusers](https://huggingface.co/docs/diffusers/index), [GLIGEN](https://github.com/gligen/GLIGEN), and [layout-guidance](https://github.com/silent-chen/layout-guidance). This code also has an implementation of [boxdiff](https://github.com/showlab/BoxDiff) and [MultiDiffusion (region control)](https://github.com/omerbt/MultiDiffusion/tree/master). Using their code means adhering to their license.

## Citation
If you use this work or the implementation in this repo, please use the citation of the original work.
```
@article{lian2023llmgrounded,
    title={LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models}, 
    author={Lian, Long and Li, Boyi and Yala, Adam and Darrell, Trevor},
    journal={arXiv preprint arXiv:2305.13655},
    year={2023}
}
```
