# From Reasoning to Recommendation: LLM-Driven Graph Attribute Augmentation for Collaborative Filtering

Welcome to the official repository of **From Reasoning to Recommendation: LLM-Driven Graph Attribute Augmentation for Collaborative Filtering** . This framework provides a suite of models and tools for graph-based machine learning, including various Graph Convolution Networks (GCNs) and Contrastive Learning methods.

## Requirements

Before running the experiments, make sure you have the following dependencies installed:

* **Python** (version >= 3.7)
* **CUDA** (for GPU acceleration; make sure you have the correct version for your setup)
* **Conda** (for managing environments)

## Setup Instructions

To reproduce the results from our experiments, follow these steps:

1. **Create a Conda Environment**

   First, create a new Conda environment using the provided `agcf` file. This file includes all the necessary dependencies to run the experiments.

2. **Download Data**

   Next, download all data through [Google Drive](https://drive.google.com/file/d/14jkM6KZwdjNpGouEy74NVH50X3Ga-C-v/view?usp=drive_link), and unzip the agcf_data.zip. The extracted directory should have the following structure:


├── data/
│   ├── amazon/
│   │   ├── attr_edges.pkl
│   │   └── ...
│   └── office/
│       └── ...
├── encoder/
└── generation/


3. **Run the Encoder Training**

   To train the encoder with a specific model and dataset, run the following command:
   ```bash
   python encoder/train_encoder --model {model_name} --dataset {dataset_name}
   ```

   * Replace `{model_name}` with the name of the model you want to use. Supported models include:

     * `lightgcn_rgcn`
     * `sgl_rgcn`
     * `simgcl_rgcn`
     * Other baseline models can be found in the `encoder/config/modelconf` folder.

   * Replace `{dataset_name}` with the dataset you want to use. Supported datasets include Amazon-Book, Yelp, Amazon-Office.
