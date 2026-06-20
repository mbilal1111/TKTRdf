# TKTRdf Benchmark: RDF Event–State Transition Representations

This repository accompanies the paper:

**"A Controlled Benchmark of RDF Event–State Transition Representations in Energy-Centric CPHSs: Structural Effects on Link and Chain Prediction"**

---

## Overview

This repository provides a **fully reproducible benchmark** for evaluating RDF-based graph representations of event–state transitions under controlled conditions.

The benchmark compares three RDF modeling strategies:

- RDF Reification  
- Canonical N-ary Relations  
- TKTRdf (Tacit Knowledge Tree RDF)

All representations are constructed such that they:

- encode identical transition semantics  
- use identical node sets  
- contain identical triple counts  

This ensures that observed differences in learning performance are attributable **only to structural properties** of the graph.

---

## Key Contributions

- Controlled benchmark for RDF-based representations  
- Structurally constrained RDF profile (TKTRdf)  
- Transition-chain prediction task for sequential reasoning  
- Evaluation across 9 KGE models  
- Comparison using both:
  - triple-level splits  
  - leakage-resistant event-level splits  

---

## Repository Structure

```
tktrdf-benchmark/
│
├── run_all.sh              ← ✅ One-command entry point
├── run_experiment.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── smart_building/
│   └── smart_grid/
│
├── scripts/
│   ├── preprocessing/
│   ├── splits/
│   └── evaluation/
│
├── results/
│   ├── raw/
│   ├── aggregate/
│   └── figures/
│
├── tables/
│
└── notebooks/
```

---

## Data

The repository includes two datasets:

- **Smart Building**
- **Smart Grid**

Each dataset is provided in three RDF representations:

- `tktrdf/`
- `reification/`
- `nary/`

Each representation uses the same:

- entities  
- relations  
- transition semantics  

---

## Dataset Format

Triples are stored as:

```
head relation tail
```

Example:

```
Sensor1 detects Room101
HVACUnit1 priorState CoolingOff
HVACUnit1 resultingState CoolingOn
```

---

## ✅ One-Command Reproducibility

To reproduce all experiments and results:

```bash
bash run_all.sh
```

This will:

1. Install dependencies  
2. Generate dataset splits (triple-level + event-level)  
3. Train all KGE models  
4. Evaluate transition-chain prediction  
5. Produce final results  

Outputs will be available in:

- `results/`
- `tables/`

---

## Manual Execution (Optional)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Generate splits

```bash
python scripts/splits/create_triple_split.py
python scripts/splits/create_event_split.py
```

### Run experiments

```bash
python run_experiment.py
```

### Evaluate results

```bash
python scripts/evaluation/evaluate_chain_prediction.py
```

---

## Evaluation Tasks

### 1. Generic Link Prediction

Standard prediction of missing triples using:

- Mean Reciprocal Rank (MRR)
- Mean Rank (MR)
- Hits@K

---

### 2. Transition-Chain Prediction

A task where models predict the **next transition in a sequence**, evaluating:

- sequential reasoning  
- transition structure learning  

---

## Results

Results are stored in:

```
results/
```

Including:

- raw model outputs  
- aggregated metrics  
- figures used in the paper  

---

## Table Mapping

| Paper Table | File |
|------------|------|
| Table IV (MRR) | tables/table_mrr.csv |
| Table V (MR) | tables/table_mr.csv |
| Table VI (Hits SB) | tables/table_hits_sb.csv |
| Table VII (Hits SG) | tables/table_hits_sg.csv |
| Table VIII | tables/table_aggregate.csv |

---

## Key Findings

- RDF Reification performs strongest on:
  - **MRR**
  - **MR**
- TKTRdf performs strongest on:
  - **Hits@K (especially H@10)**
- Representation performance is:
  - **metric-dependent**
  - **evaluation-dependent**

These results show that structural regularity improves **ranking robustness**, but does not uniformly improve exact-ranking performance.

---

## Implementation Notes

- All models share identical hyperparameters across representations  
- Graph construction is fully controlled to eliminate confounding factors  
- Event-level splits prevent leakage between training and test sets  
- All experiments use a fixed random seed (42) for reproducibility  

---

## License

This repository is released under the MIT License.

---

## Citation

If you use this work, please cite:

```
@article{bilal2026tktrdf,
  title={A Controlled Benchmark of RDF Event–State Transition Representations in Energy-Centric CPHSs},
  author={Bilal, Mohammad and others},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2026}
}
```

---

## Contact

- Mohammad Bilal  
- TU Wien – Automation Systems  
- mohammad.bilal@tuwien.ac.at  

---

## Acknowledgements

This work is supported by the SENSE project and contributions from domain experts in smart building and smart grid systems.

---
