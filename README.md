# 💧 AQUA Watermark: Copyright Protection for Multimodal Knowledge in RAG-as-a-Service

[![Paper Status](https://img.shields.io/badge/Status-arXiv%20Preprint%20--%20Under%20Review-blue)](https://arxiv.org/abs/YOUR_ARXIV_ID)  [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) 

## ⚙️ Environment Setup


1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tychenn/AQUA.git
    cd AQUA
    ```
2.  **Create and activate a Conda environment (recommended):**
    ```bash
    conda env create -f environment.yml
    conda activate AQUA
    ```

## 📊 Datasets 

* [**MMQA**]([https://github.com/allenai/multimodalqa])  
* [**WebQA**]([https://github.com/WebQnA/WebQA])  
* Place the downloaded data in the path: `datasets/MMQA/images` and `datasets/WebQA/images`
* According to the format of the uploaded sample data, place the generated watermark images and their probe queries in the correct location.

    
    
## 🚀 Reproduce experimental results


1.  **Forming the database**
    ```bash
    python -m utils.indexing_faiss
    ```
    * You may need to change the location of the target and source folders.

2.  **Get experimental results**
    * Run the `.py` file in the `experiments` folder to get the corresponding experimental results, taking the table data in the effectiveness section as an example:
    ```bash
    python -m experiments.effectiveness.table
    ```
    * You may need to modify the parameters in the file.

## Contact me
If you have any questions about reproducing the code, you can raise them in the issue section, or contact me through my email after the paper review is completed.


## Citation

If you find our work helpful, please consider citing our paper:

```bibtex
