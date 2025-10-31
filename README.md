# TKTRdf
 TKTRdf, a taxonomy–driven RDF extension 

 # SENSE Project: Knowledge Graph Evaluation for Explainable Cyber-Physical Systems

This repository hosts the code and data used in the **SENSE Project** ([https://sense-project.net](https://sense-project.net)), focusing on Explainable Cyber-Physical Systems (ExpCPS).  
The goal is to evaluate knowledge graph embedding models on data from Smart Building and Smart Grid use cases to improve explainability, efficiency, and sustainability.

The evaluation compares three knowledge graph (KG) variants:

- Standard RDF  
- Reified RDF  
- TKTRdf (causality-focused)

We apply embedding models TransE, TransH, and ComplEx to these KGs, measuring performance using link prediction metrics with confidence intervals. The evaluation is performed on two real-world CPS use cases: **Smart Building** and **Smart Grid**, capturing the complexity of modern CPS environments.

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

├── Quality metrics script.py # Compute KG structural quality metrics
├── Working script wit transe, transh, complex, ci and precision.py # Main embedding evaluation pipeline
├── Results 301025.xlsx # Sample or final results summary
├── SENSEsystemdataSB.ttl # RDF KG for Smart Building
├── systemdatasb_reified.ttl # Reified RDF KG for Smart Building
├── SmartbuildingKG.xlsx # TKTRdf (Excel) KG for Smart Building
├── SENSEsystemdataSG.ttl # RDF KG for Smart Grid
├── systemdatasg_reified.ttl # Reified RDF KG for Smart Grid
├── SmartgridKG.xlsx # TKTRdf (Excel) KG for Smart Grid
├── README.md # This file


## Code Overview

The main script `working_file....py`:
1. Reads and converts knowledge graph data from Excel and RDF.
2. Convert RDF into reified RDF to enable metadata on statements.  
3. Generates triples for model training and testing.
4. Runs PyKEEN pipelines across several embedding models.
5. Computes evaluation metrics and confidence intervals.
6. Exports results to an Excel summary.

## Built With

- Python 3.x  
- PyKEEN  
- pandas  
- numpy  
- rdflib  
- scikit-learn  
- openpyxl
- networkx 

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


This performs embedding training and evaluation on RDF, Reified RDF, and TKTRdf graphs across embedding models. Results with confidence intervals are saved to `kg_evaluation_with_ci.xlsx`.

## Results
The evaluation comprises two main result types: 
These are reported separately for RDF, Reified RDF, and TKTRdf graphs, showing the influence of causal semantics and reification on embedding performance and explainability.


**1. Embedding Evaluation Results**
Performance metrics such as:

Mean Reciprocal Rank (MRR)

Mean Rank (MR)

Hits@10

Transition Precision & Recall

are computed for each embedding model (TransE, TransH, ComplEx) across the three KG variants (RDF, Reified RDF, TKTRdf). These metrics show the predictive quality and explainability of event–state relationships. The results with calculated 95% confidence intervals are saved in:

kg_evaluation_with_ci.xlsx

This file details model performance per KG type and use case, enabling comparative analysis.

**2. Graph Quality Metrics**
Structural quality metrics provide insight into the underlying knowledge graph topology and content, influencing model training and interpretability. These include:

Coverage (Entities and Relations): How many unique entities and relations the KG contains, indicating breadth of domain representation.

Average Degree: The typical connectivity of entities, showing graph density.

Connected Components: The number of isolated subgraphs, reflecting graph fragmentation or modularity.

Degree and Relation Standard Deviations: The variability of connections per entity and frequency of relation types, showing graph heterogeneity or balance.

Clustering Coefficient: The tendency of nodes to form tightly connected groups, indicating graph transitivity and complexity.

These metrics are computed for each KG variant and are saved in:

kg_quality_metrics_sg.xlsx

Together, these quality metrics complement embedding evaluation by highlighting structural differences between RDF, Reified RDF, and TKTRdf graphs, helping interpret the impact of causal semantics and graph design on link prediction performance.


## Results

The evaluation table shows MRR, MR, and Hits@10 values along with transition precision and recall for both knowledge graph types (RDF and Excel).  
Results are stored in the Excel file with the following structure:

| Metric | TransE RDF | TransE Reified |  TransE TKTrdf | TransH RDF |  TransH Reified | TransH TKTrdf | ComplEx RDF | | ComplEx Reified | ComplEx TKTrdf |
|---------|-------------|---------------|-------------|---------------|--------------|----------------||---------------|-----------------||---------------|

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


