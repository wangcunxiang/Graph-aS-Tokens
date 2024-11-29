<div align="center">
  <h1> Graph-aS-Tokens </h1>
  
  [![License: Apache-2.0](https://img.shields.io/crates/l/Ap?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
</div>

  # 📌 Table of Contents
- [Introduction](#-introduction)
- [Development](#-development)
- [Data](#-data)
- [License](#-license)
- [Citation](#-citation)
- [Contact](#-contact)
  
# 🚀 Introduction
This repository is for the ACL-2023 findings paper "Exploiting Abstract Meaning Representation for Open-Domain Question Answering" [paper](https://aclanthology.org/2023.findings-acl.131/).

# 💻 Development
## Setup
Clone the repository from GitHub and install:
```
git clone https://github.com/wangcunxiang/Graph-aS-Tokens.git
cd Graph-aS-Tokens/reader/
pip install -e ./
```
## Train Script
### Reading
```
name={name}
CUDA_VISIBLE_DEVICES=X python train_reader.py \
        --train_data path/to/train_file \
        --eval_data path/to/dev_file \
        --model_size [large]/[base] \
        --per_gpu_train_batch_size 1 \
        --per_gpu_eval_batch_size 1 \
        --accumulation_steps 64 \
        --total_steps 320000 \
        --eval_freq 10000 \
        --save_freq 10000 \
        --n_context 10 \
        --text_maxlength 200 \
        --graph_maxlength 310 \
        --seed 0 \
        --name ${name} \
        --checkpoint_dir checkpoint \
        --graph_as_token

```

### Reranking
```
name={name}
CUDA_VISIBLE_DEVICES=X python train_reranker.py \
        --train \
        --is_amr \
       --train_data path/to/train_file \
        --eval_data path/to/dev_file \
        --psgs ../NQ/psgs_nq.json \
        --model bart \
        --loss cross_entropy \
        --num_negative_psg 7 \
        --train_bsz 4 \
        --lr 1e-5 \
        --epoch 10 \
        --mhits_bar 10 \
        --node_length 140 \
        --edge_length 170   \
        --note name
```
## Validation Script
### Reading
```
name={name}
CUDA_VISIBLE_DEVICES=X python test_reader.py \
        --eval_data path/to/test_file \
        --model_path path/to/checkpoint_dir/ \
        --per_gpu_eval_batch_size 10 \
        --n_context 10 \
        --write_results \
        --answer_maxlength 30 \
        --text_maxlength 200 \
        --graph_maxlength 310 \
        --seed 0 \
        --name ${name} \
        --checkpoint_dir checkpoint \
        --graph_as_token \
        --write_results
```
### Reranking
```
name={name}
CUDA_VISIBLE_DEVICES=X python train_reranker.py \
        --val \
        --is_amr \
        --eval_data path/to/dev_file \
        --psgs ../NQ/psgs_nq.json \
        --model bart \
        --loss cross_entropy \
        --num_negative_psg 7 \
        --train_bsz 4 \
        --lr 1e-5 \
        --ckpt N \
        --mhits_bar 10 \
        --node_length 140 \
        --edge_length 170   \
        --note name
```

<!-- ## Checkpoints

The best-performed RFiD checkpoint on Natural Questions is [here](https://drive.google.com/file/d/1q7UC2rFxtxb3dM0946dvJnWxm4ljTge1/view?usp=sharing). -->

# 📝 Data
### Following [FiD](https://github.com/facebookresearch/FiD), we use the same data and format and parse them into AMR graphs. However, due to the policy of Amazon company, we are unable to release the parsing code, but we can share that it is implemented based on the [AMRBART](https://huggingface.co/xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing) model.
## Download data
Though we cannot share the parsing code, we can share the parsed data, which can be accessed [here](https://drive.google.com/file/d/1qCrmV_zKrjHZBfbw6uDCMI83Co2GEUfA/view?usp=sharing).


## Data format
The expected data format is a list of entry examples, where each entry example is a dictionary containing
```
id: example id, optional
question: question text
answers: list of answer text for training/evaluation
ctxs: a list of passages where each item is a dictionary containing - title: article title - text: passage text - graph
```
Entry example:
```
{
  'id': '0',
  'question': 'who got the first nobel prize in physics?',
  'answers': ["Wilhelm Conrad R\u00f6ntgen"],
  'ctxs': [
            {
                "id": "628713",
                "title": "Nobel Prize in Physics",
                "text": "Nobel Prize in Physics The Nobel Prize in Physics () is a yearly award given by the Royal Swedish Academy of Sciences for those who have made the most outstanding contributions for mankind in the field of physics. It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine. The first Nobel Prize in Physics was awarded to physicist Wilhelm Röntgen in recognition of the extraordinary services he",
                "nodes": ["multi-sentence", "question", "get", "amr-unknown", "prize", "Nobel Memorial Prize in Physics", "name", "Nobel", "Prize", "in", "Physics", "ordinal-entity", "1", "context", "award", "organization", "Royal Swedish Academy of Sciences", "mean", "prize", "-", "name", "\"The", "-", "Nobel-Class", "-", "-", "Physics:", "op6", "person", "contribute", "mankind", "have-degree", "outstanding", "most", "field", "award", "rate-entity", "temporal-quantity", "1", "year", "prize", "\"The", "include", "prize", "5", "The Nobel Prize", "include", "and", "prize", "Chemistry Nobel Prize,", "prize", "award", "Pulitzer Prize", "name", "1984", "Literature", "peace", "award", "-", "ordinal-entity", "-1", "Nobel", "Prize-in", "snt6", "prep-or", "op7", "since", "date-entity", "1901", "establish", "will", "person", "Alfred Nobel", "date-entity", "1895", "prize", "Award", "prize", "Nobel", "ordinal-entity", "-1", "from", "-", "Wilhelm R\u00c3\u00b6ntgen", "Wilhelm", "R\u00c3\u00b6twgen", "temporal-quantity", "2", "year", "back"],
                "edges": [[0, "snt1", 1], [1, "ARG1", 2], [2, "ARG0", 3], [2, "ARG1", 4], [4, "wiki", 5], [4, "name", 6], [6, "op1", 7], [6, "op2", 8], [6, "op3", 9], [6, "op4", 10], [4, "ord", 11], [11, "value", 12], [1, "topic", 13], [13, "mod", 4], [0, "snt2", 14], [14, "ARG0", 15], [15, "wiki", 16], [15, "name", 6], [15, "ARG1-of", 17], [17, "ARG2", 18], [18, "wiki", 19], [18, "name", 20], [20, "op1", 21], [20, "op2", 22], [20, "op2", 23], [20, "op3", 24], [20, "op4", 25], [20, "op5", 26], [20, "", 27], [17, "ARG1-of", 17], [14, "ARG2", 28], [28, "ARG0-of", 29], [29, "ARG2", 30], [29, "ARG1-of", 31], [29, "ARG2", 32], [29, "ARG3", 33], [28, "mod", 34], [34, "mod", 34], [14, "ARG3-of", 35], [35, "frequency", 36], [36, "ARG3", 37], [37, "quant", 38], [37, "unit", 39], [36, "ARG4", 40], [40, "wiki", 41], [40, "name", 11], [40, "ARG1-of", 39], [40, "ARG1-of", 11], [40, "ARG2-of", 42], [42, "ARG2", 43], [43, "quant", 44], [43, "wiki", 45], [43, "name", 35], [43, "ARG1-of", 35], [43, "ARG2", 46], [46, "ARG2", 34], [46, "ARG3", 47], [47, "op1", 48], [48, "wiki", 49], [48, "name", 2], [48, "ARG1-of", 40], [47, "op2", 50], [50, "wiki", 51], [50, "wiki", 52], [50, "unit", 53], [53, "op1", 54], [53, "op2", 55], [50, "op3", 56], [50, "op4", 57], [57, "wiki", 58], [57, "ord", 59], [59, "value", 60], [57, "domain", 61], [61, "op5", 62], [61, "", 63], [61, "", 64], [61, "", 65], [57, "ARG1", 14], [47, "op3", 15], [43, "time", 66], [66, "op1", 67], [67, "year", 68], [43, "manner", 69], [69, "ARG0", 70], [70, "poss", 71], [71, "wiki", 72], [71, "name", 17], [70, "time", 73], [70, "year", 74], [43, "op2", 61], [43, "op3", 75], [75, "wiki", 76], [75, "name", 15], [75, "ARG1-of", 46], [75, "ARG1-of", 42], [42, "op4", 75], [42, "op5", 77], [77, "wiki", 78], [77, "ord", 79], [79, "value", 80], [77, "ARG1", 81], [77, "ARG2", 67], [40, "domain", 61], [35, "op5", 69], [35, "ARG1", 4], [35, "ARG2", 70], [35, "wiki", 82], [35, "wiki", 83], [35, "name", 3], [35, "name", 71], [35, "op2", 84], [35, "op2-of", 85], [14, "mod", 34], [14, "ARG0-of", 29], [0, "ARG3", 34], [0, "ARG1", 75], [0, "ARG0", 70], [0, "mod", 77], [86, "quant", 87], [86, "unit", 88], [86, "mod", 6]]
            },
            {
                "id": "284495",
                "title": "Nobel Prize",
                "text": "His son, George Paget Thomson, received the same prize in 1937 for showing that they also have the properties of waves. William Henry Bragg and his son, William Lawrence Bragg, shared the Physics Prize in 1915 for inventing the X-ray spectrometer. Niels Bohr was awarded the Physics prize in 1922, as was his son, Aage Bohr, in 1975. Manne Siegbahn, who received the Physics Prize in 1924, was the father of Kai Siegbahn, who received the Physics Prize in 1981. Hans von Euler-Chelpin, who received the Chemistry Prize in 1929, was the father of Ulf von Euler, who was awarded",
                "nodes": ["multi-sentence", "question", "person", "get", "prize", "Nobel Memorial Prize in Physics", "name", "Nobel", "Prize", "in", "Physics", "ordinal-entity", "1", "physics", "prize", "-", "name", "1984", "Nobel-prize", "mean", "receive", "person", "George P. Thomson", "name", "George", "Paget", "2", "Thomson", "have-rel-role", "he", "son", "show", "have", "they", "property", "wave", "also", "date-entity", "1937", "share", "person", "-", "William Henry Bragg", "name", "William", "Henry", "1", "Bragg,", "person", "-", "invent", "spectrometer", "xray", "op6", "award", "prize", "\"Physics", "date-entity", "1915", "award", "person", "person", "formulae", "Niels Bohr", "physics", "1922", "snt7", "award", "person", "Arne Claus Bohr (chemist)", "name", "\"Aage", "-Bohr\"", "date-entity", "1975", "have-rel-role", "person", "Manne", "receive", "prize", "+", "name", "1923", "Physics-Specialist", "+", "-Livert", "father", "person", "person", "Kai", "name", "-", "Kai", "-Siegbahn-", "from", "1981", "expressive", "+", "Hans", "+", "von", "+", "Euler-Chelpin", "Ulf von Euler", "2", "award", "person", "physicist", "name", "Hundert", "1", "chemistry"],
                "edges": [[0, "snt1", 1], [1, "ARG1", 2], [2, "ARG0-of", 3], [3, "ARG1", 4], [4, "wiki", 5], [4, "name", 6], [6, "op1", 7], [6, "op2", 8], [6, "op3", 9], [6, "op4", 10], [4, "ord", 11], [11, "value", 12], [4, "mod", 13], [4, "mod", 4], [4, "mod", 14], [14, "wiki", 15], [14, "name", 16], [16, "op1", 17], [16, "op2", 18], [2, "ARG1-of", 19], [19, "ARG2", 20], [20, "ARG0", 21], [21, "wiki", 22], [21, "name", 23], [23, "op1", 24], [23, "op2-of", 25], [23, "op3", 26], [23, "op4", 27], [21, "ARG0-of", 28], [21, "ARG1", 29], [21, "ARG2", 30], [20, "ARG2", 4], [20, "ARG3", 31], [31, "ARG0", 21], [31, "ARG1", 32], [32, "ARG0", 33], [32, "ARG1", 34], [34, "poss", 35], [32, "mod", 36], [31, "ARG3", 4], [20, "time", 37], [37, "year", 38], [19, "snt3", 39], [39, "ARG0", 40], [40, "wiki", 41], [40, "wiki", 42], [40, "name", 37], [40, "name", 43], [43, "op2", 44], [43, "op3", 45], [43, "op3", 46], [43, "op4", 47], [39, "op2", 48], [48, "wiki", 49], [48, "name", 37], [48, "ARG1-of", 37], [48, "ARG0-of", 13], [19, "time", 37], [19, "ARG1", 50], [50, "ARG0", 40], [50, "ARG1", 51], [51, "mod", 52], [50, "ARG2", 52], [2, "", 53], [54, "ARG1", 55], [55, "wiki", 56], [55, "name", 11], [54, "ARG2", 52], [54, "time", 57], [57, "year", 58], [2, "snt4", 59], [59, "ARG1", 55], [59, "ARG2", 60], [60, "wiki", 61], [61, "wiki", 62], [61, "name", 2], [60, "wiki", 63], [60, "name", 4], [59, "ARG3", 64], [59, "year", 65], [2, "", 66], [67, "ARG2", 68], [68, "wiki", 69], [68, "name", 70], [68, "op1", 71], [68, "op2", 72], [67, "ARG3", 64], [67, "time", 73], [73, "year", 74], [2, "snt5", 75], [75, "ARG0", 76], [76, "wiki", 77], [76, "name", 19], [76, "ARG1-of", 3], [76, "ARG2-of", 78], [78, "ARG1", 79], [79, "wiki", 80], [79, "name", 81], [81, "op1", 82], [81, "op2", 83], [81, "op3", 84], [81, "op4", 85], [79, "domain", 55], [76, "ARG4", 86], [75, "accompanier", 87], [87, "wiki", 88], [87, "wiki", 89], [87, "unit", 90], [90, "op1", 91], [90, "op2", 92], [90, "op3", 93], [87, "ARG1", 94], [87, "ARG0", 64], [87, "ARG1", 55], [87, "time", 64], [87, "year", 95], [2, "consist-of", 67], [2, "mode", 96], [2, "name", 68], [2, "op1", 97], [2, "op2", 98], [2, "op2", 99], [2, "op3", 100], [2, "op4", 101], [2, "op5", 102], [1, "source-of", 73], [1, "wiki-of", 103], [1, "name-of", 75], [1, "op2", 104], [0, "ARG0", 105], [106, "wiki", 107], [106, "name", 108], [108, "op1-of", 109], [108, "op2", 110]]
            }
          ]
}
```
# 📜 License

This repository is released under the [Apache-2.0 License](LICENSE).

# 📚 Citation

If you find this repository useful, please cite it as follows:
```bibtex
@inproceedings{GST,
    title = "Exploiting {A}bstract {M}eaning {R}epresentation for Open-Domain Question Answering",
    author = "Wang, Cunxiang  and
      Xu, Zhikun  and
      Guo, Qipeng  and
      Hu, Xiangkun  and
      Bai, Xuefeng  and
      Zhang, Zheng  and
      Zhang, Yue",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.131",
    doi = "10.18653/v1/2023.findings-acl.131",
    pages = "2083--2096",
    abstract = "The Open-Domain Question Answering (ODQA) task involves retrieving and subsequently generating answers from fine-grained relevant passages within a database. Current systems leverage Pretrained Language Models (PLMs) to model the relationship between questions and passages. However, the diversity in surface form expressions can hinder the model{'}s ability to capture accurate correlations, especially within complex contexts. Therefore, we utilize Abstract Meaning Representation (AMR) graphs to assist the model in understanding complex semantic information. We introduce a method known as Graph-as-Token (GST) to incorporate AMRs into PLMs. Results from Natural Questions (NQ) and TriviaQA (TQ) demonstrate that our GST method can significantly improve performance, resulting in up to 2.44/3.17 Exact Match score improvements on NQ/TQ respectively. Furthermore, our method enhances robustness and outperforms alternative Graph Neural Network (GNN) methods for integrating AMRs. To the best of our knowledge, we are the first to employ semantic graphs in ODQA.",
}
```
## 📮 Contact
If you have any questions or feedback, please feel free to reach out at wangcunxiang@westlake.edu.cn.
