Data-driven discovery of linear molecular probes with optimal selective affinity for PFAS in water
--

This repository demonstrates the data-driven discovery of linear molecular probes with optimal selective affinity for per- and polyfluoroalkyl substances (PFAS) in water. It integrates molecular dynamics (MD) simulations, enhanced sampling methods, deep representational learning via variational autoenecoders, surrogate model training and multi-objective Bayesian optimization using random scalarations. We use perfluorooctanesulfonic acid (PFOS) as the target PFAS and sodium dodecyl sulfate (SDS) as a representative interfernt to demonstrate our approach. 

<p align="center">
<img width="843" alt="Screenshot 2023-08-22 at 11 00 17 AM" src="https://github.com/Ferg-Lab/activeLearningPFASLinear/assets/38693318/083976aa-eb17-43b8-b232-90b9bfeb7218">
</p>

---

Installation

`INSTALL_LOCATION=<set path here for installation of conda env>`

`conda env create -f environment.yml --prefix $INSTALL_LOCATION` 

---

Usage

1. [Codes](./Codes): Estimation of binding free energies & binding constants. Please see [Codes/README.md](./Codes/README.md) for getting started.
2. [Data](./Data): GROMACS simulation files and PLUMED template file (master-pbmd-files-final) for performing enhanced sampling, probes embedding in latent space and helper data (VAE_data), JCED Supplementary Information data (JCED_data_each_cycle), binding free energy and binding constant all cycles combined and in each cycle (GPR_training_data),and probe structures in each cycle (smiles-each-cycle). Please see [Data/README.md](./Data/README.md).
3. [Notebooks](./Notebooks): Analysis python notebooks for calculating potential of mean force (PMF) profiles, binding free energies & binding constants, GPR (Gaussian process regression) training, multi-objective Bayesian optimization using random scalarizations, and example notebook ([Notebooks/example_notebook_to_read_SI_tables_deltaG_Kb.ipynb](./Notebooks/example_notebook_to_read_SI_tables_deltaG_Kb.ipynb)) to read data reported in Supplementary Information of JCED ([Data/JCED_data_each_cycle](./Data/JCED_data_each_cycle)). Please see [Notebooks/README.md](./Notebooks/README.md).
   
---

Cite

If you use the codes or notebooks from this repo in your work, please cite:

S. Dasetty, M. Topel, Y. Tang, Y. Wang, E. Jonas, S. Darling, J. Chen, A. L. Ferguson. "Data-driven discovery of linear molecular probes with optimal selective affinity for PFAS in water" XXXXX. DOI: XXXX

```
@article{ferglab2023PFAS,
  title={Data-driven discovery of linear molecular probes with optimal selective affinity for PFAS in water},
  author={Dasetty, S. and Topel, M. and Tang, Y. and Wang, Y. and Darling, S. and Chen, J. and Ferguson, A.L.},
  journal={XXXX},
  volume={XXXX},
  number={XXXX},
  pages={XXXX-XXXX},
  year={2023},
  publisher={XXXX}
}
```


