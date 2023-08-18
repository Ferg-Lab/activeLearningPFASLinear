Data-driven discovery of linear molecular probes with optimal selective affinity for PFAS in water
--

This repository demonstrates the data-driven discovery of linear molecular probes with optimal selective affinity for per- and polyfluoroalkyl substances (PFAS) in water. It integrates molecular dynamics (MD) simulations, enhanced sampling methods, deep representational learning via variational autoenecoders, surrogate model training and multi-objective Bayesian optimization using random scalarations. We use perfluorooctanesulfonic acid (PFOS) as the target PFAS and sodium dodecyl sulfate (SDS) as a representative interfernt to demonstrate our approach. 

---

Usage

1. [Codes](./Codes): Estimation of binding free energies & binding constants. Please see [Codes/README.md](./Codes/README.md) for getting started.
2. [JCED_data_each_cycle](./JCED_data_each_cycle): Sensitivity & selectivity of each probe in each cycle for PFOS relative to SDS.
3. [Notebooks](./Notebooks): Analysis python notebooks for calculating potential of mean force (PMF) profiles, binding free energies & binding constants, VAE (variational autoencoder model) model, GPR (Gaussian process regression) training, and multi-objective Bayesian optimization using random scalarizations. Please see [Notebooks/README.md](./Notebooks/README.md) to get started.
   
---

Cite

If you use the codes/notebooks in your work, please cite:

S. Dasetty, M. Topel, Y. Tang, Y. Wang, E. Jonas, S. Darling, J. Chen, A. L. Ferguson. "Data-driven discovery of linear molecular probes with optimal selective affinity for PFAS in water" XXXXX. DOI: XXXX

```
@article{ferglab2023PFAS,
  title={Data-driven discovery of linear molecular probes with optimal selective affinity for PFAS in water},
  author={Dasetty, S. and Topel, M. and Tang, Y. and Wang, Y. and Darling, S. and Chen, J. and Ferguson, A.L.},
  journal={XXXX},
  volume={XX},
  number={XX},
  pages={XX-XX},
  year={2023},
  publisher={XXXX}
}
```


