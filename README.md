# TKTRdf
 TKTRdf, a taxonomy–driven RDF extension 

 # SENSE Project: Knowledge Graph Evaluation for Explainable Cyber-Physical Systems

This repository hosts the code and data used in the **SENSE Project** ([https://sense-project.net](https://sense-project.net)), focusing on Explainable Cyber-Physical Systems (ExpCPS).  
The goal is to evaluate knowledge graph embedding models on data from Smart Building and Smart Grid use cases to improve explainability, efficiency, and sustainability.

![Knowledge Graph Evaluation Pipeline](sense_pipeline.png)

## Table of Contents
- [About the Project](#about-the-project)
- [Research Context](#research-context)
- [Repository Structure](#repository-structure)
- [Code Overview](#code-overview)
- [Built With](#built-with)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## About the Project

This repository provides an experimental pipeline for evaluating **knowledge graph embedding models** using the [PyKEEN](https://github.com/pykeen/pykeen) library.  
The evaluation compares models (TransE, TransH, ComplEx) across two types of graph data:
- Triples extracted from **Excel-based datasets**.
- Triples parsed from **RDF semantic files**.

Each model’s performance is measured using:
- Mean Reciprocal Rank (MRR)
- Mean Rank (MR)
- Hits@10
- Transition Precision & Recall

Confidence intervals for each metric are computed from multiple training runs.

## Research Context

Cyber-Physical Systems (CPS) such as transportation networks, smart energy grids, and intelligent buildings are increasingly complex and dynamic.  
Their explainability — the ability to understand *why* certain events occur — is decreasing, reducing transparency and operational safety.

The **SENSE Project** advances ExpCPS research in three ways:
1. Extending Digital Twin architectures.
2. Developing semantics-based explainability mechanisms.
3. Creating personalized, interactive interfaces for system users.

This repository supports those efforts by building **knowledge graph evaluation pipelines** for modeling and interpreting CPS behavior.

## Repository Structure

├── data/
│ ├── SmartbuildingKG.xlsx
│ ├── SENSEsystemdatasb.ttl
│ ├── SmartgridKG.xlsx
│ ├── SENSEsystemdatasg.ttl
├── code/
│ └── kg_evaluation.py
├── results/
│ └── kg_evaluation_with_ci.xlsx
├── README.md


## Code Overview

The main script `kg_evaluation.py`:
1. Reads and converts knowledge graph data from Excel and RDF.
2. Generates triples for model training and testing.
3. Runs PyKEEN pipelines across several embedding models.
4. Computes evaluation metrics and confidence intervals.
5. Exports results to an Excel summary.

## Built With

- Python 3.x  
- PyKEEN  
- pandas  
- numpy  
- rdflib  
- scikit-learn  
- openpyxl  

## Getting Started

### Prerequisites

Install dependencies:
pip install pandas numpy pykeen rdflib scikit-learn openpyxl


### Installation

1. Clone the repository:
git clone https://github.com/mbilal1111/TKTRdf
cd TKTRdf


2. Place your data files in the `https://github.com/mbilal1111/TKTRdf/` folder.

3. Run the pipeline:
python code/kg_evaluation.py


## Usage

The script executes multiple training runs per model, computes performance metrics, and saves results to:
results/kg_evaluation_with_ci.xlsx (you will need to change the location of the output file or the name so the next time the script is run with a new use case the file is not overwritten)


Each output file includes metrics and 95% confidence intervals across models:
- TransE
- TransH
- ComplEx

## Results

The evaluation table shows MRR, MR, and Hits@10 values along with transition precision and recall for both knowledge graph types (RDF and Excel).  
Results are stored in the Excel file with the following structure:

| Metric | TransE RDF | TransE TKTrdf | TransH RDF | TransH TKTrdf | ComplEx RDF | ComplEx TKTrdf |
|---------|-------------|---------------|-------------|---------------|--------------|----------------|

## License

This repository is released under the **No License**.  
See `LICENSE` for details.

## Contact

Mohammad Bilal
Institute of Computer Engineering, Research Unit Automation Systems, TU Wien, 1040, Vienna, Austria 
Email: mohammad.bilal@tuwien.ac.at

Project Page: [https://sense-project.net](https://sense-project.net)

## Acknowledgements

This work is part of the **SENSE Project**, which investigates Explainable Cyber-Physical Systems (ExpCPS) as part of Austria’s smart infrastructure innovation initiative.

SENSE contributes to:
- Sustainable operation of smart buildings and energy grids.  
- Human-centered interfaces for explainable systems.  
- Enhanced transparency in CPS decision-making.

---


