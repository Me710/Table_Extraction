
Table Extraction Project

Welcome to the Table Extraction Project! This internship experience has been an exciting journey that has deepened our understanding of data science and AI while focusing on two crucial components:

Table Detection Model with Table Transformers: In this project, we delved into cutting-edge techniques aimed at enhancing table detection within various types of documents. Leveraging the remarkable capabilities of Table Transformers, we aimed to improve the accuracy and efficiency of table recognition, a fundamental task in data extraction. Notably, we fine-tuned the Microsoft DETR model on two diverse datasets: TNCR and FoodUM.

Post-Processing for Enhanced Extraction: Beyond table detection, our project went the extra mile by implementing advanced post-processing methods. These refinements were designed to ensure the highest levels of accuracy and reliability in the extraction process.

What You'll Find Here:

📓 Notebooks: Explore the Jupyter notebooks we used for data analysis, model training, and evaluation. These notebooks provide detailed insights into our methodologies.

📜 Scripts for Data Preparation: Dive into the scripts we developed for data preprocessing and transformation. Understanding the data pipeline is crucial for replicating and extending our work.

🧠 Model Weights: We provide the pre-trained model weights after 30 epochs of fine-tuning. These weights are ready to use for your table extraction tasks or further training.

🌐 Front-end Deployment with FastAPI: Discover our front-end deployment using FastAPI, enabling you to interact with the model and see it in action. We've made it user-friendly and accessible for your convenience.

You can download the dataset from this github : https://github.com/abdoelsayed2016/TNCR_Dataset/tree/main/data

# Data Prepration 
You should convert your Pascal VOC XML annotation files to a single COCO Json file so we can train our models.
The Raw folder for the data should be like below, then used generateVOC2JSON.py

    .
    │   ├── data
    │   │   ├── TNDC Dataset
    │   │   │  ├── Annotations
    │   │   │  │    ├── 000ad0bb60e1ec4176713693a41f2115.xml
    │   │   │  │    ├── ...
    │   │   │  │    ├── ffda050fe78074f78b874540ad218fb9.xml
    │   │   │  ├── ImageSets
    │   │   │  │    ├── Main 
    │   │   │  │    │     ├── test.txt
    │   │   │  │    │     ├── train.txt
    │   │   │  │    │     ├── val.txt
    │   │   │  ├── Images
    │   │   │  │    ├── 000ad0bb60e1ec4176713693a41f2115.jpg
    │   │   │  │    ├── ...
    │   │   │  │    ├── ffda050fe78074f78b874540ad218fb9.jpg
    |   |   |  |    |---ordering.py
    │   │   │  ├── Output Json
    │   │   ├── generateVOC2JSON.py   




