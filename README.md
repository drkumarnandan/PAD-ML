# Molecular Determinants of α-Gliadin Dimerization and Early Aggregation

This repository contains machine-learning and data-driven analysis scripts associated with the manuscript:

**Molecular Determinants of α-Gliadin Dimerization and Early Aggregation in Hydroalcoholic Environments: A Long-Timescale Molecular Dynamics and Data-Driven Analysis**

## Description

The repository provides scripts and processed descriptor tables used to identify key molecular determinants of compact-associated α-gliadin dimer formation from molecular dynamics simulations.

The analysis focuses on:

- Interchain contact persistence
- Interchain hydrogen bonding
- COM distance
- Interface residue participation
- Hotspot-mediated contacts
- Hotspot–hotspot interactions
- Local water/ethanol coordination
- Machine-learning-based feature interpretation

## Repository Contents

- data/: Processed descriptor tables
- scripts/: Analysis and plotting scripts
- models/: Trained machine-learning models
- figures/: Publication figures generated from the analysis

## Models

The machine-learning analysis includes:

- Logistic Regression
- Random Forest
- XGBoost

The goal is feature interpretation rather than development of a standalone predictive model.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
