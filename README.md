# TriC-Motion: Tri-Domain Causal Modeling Grounded Text-to-Motion Generation (ICLR 2026)

![Intro](assets/Intro.jpg)

[![img](https://camo.githubusercontent.com/200631cfd662f27e8aca14e6631753cccd0826e6cc61cd641c6d87536fdb8b92/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d50726f6a656374266d6573736167653d5061676526636f6c6f723d726564)](https://caoyiyang1105.github.io/TriC-Motion/)[![img](https://camo.githubusercontent.com/137c48eb883230d3245afbc9eaffed00b771ded42438b2f4398c307b4f2c2b9c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f61725869762d50617065722d3c434f4c4f523e2e737667)](https://arxiv.org/abs/2602.08462v1)

## ⚙️ Getting Started

### 1. Conda Environment

```shell
conda env create -f environment.yml
```

We tested our code on `Python 3.9.12` and `PyTorch 1.10.0+cu111`.

### 2. Pre-trained Models and Dependencies

### Download Dependencies

```shell
bash prepare/download_glove.sh
bash prepare/download_smpl_files.sh
bash prepare/download_t2m_evaluator.sh
```

### Download Pre-trained Models

```shell
bash prepare/download_models.sh
```

### 3. Obtain Data

**HumanML3D** - Follow the instruction in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git), then copy the dataset to your data folder:

```shell
cp -r ./HumanML3D/ dataset/HumanML3D
```

**SnapMoGen** - Download the data from [huggingface](https://huggingface.co/datasets/Ericguo5513/SnapMoGen), then place it in the following directory:

```shell
cp -r ./SnapMoGen dataset/SnapMoGen
```

## 🏃 Generation

### (a) Generate with single textual instruction

```python
python -m sample.generate --model_path ./save/tric_motion_L/model.pt --text_prompt "A person jumps up then waits for a bit and then walks forwards."
```

### (b) Generate from a prompt file

```python
python -m sample.generate --model_path ./save/tric_motion_L/model.pt --input_text ./assets/example_text_prompts.txt
```

## 📖 Evaluation

### HumanML3D

```python
python -m eval.eval_humanml --model_path ./save/tric_motion_model_humanml/model000800000.pt
```

## 🍀 Acknowledgments

This code is standing on the shoulders of giants, we would like to thank the following contributors that our code is based on:

[guided-diffusion](https://github.com/openai/guided-diffusion), [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [MDM](https://github.com/GuyTevet/motion-diffusion-model/tree/main) and [MoGenTS](https://github.com/weihaosky/mogents).

## 👍 Citation

```txt
@inproceedings{Cao2026TriCMotionTC,
  title={TriC-Motion: Tri-Domain Causal Modeling Grounded Text-to-Motion Generation},
  author={Yiyang Cao and Yunze Deng and Ziyu Lin and Bin Feng and Xinggang Wang and Wenyu Liu and Dandan Zheng and Jingdong Chen},
  year={2026},
  url={https://api.semanticscholar.org/CorpusID:285454295}
}
```