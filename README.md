# LightMIS: Ultra-Lightweight Medical Image Segmentation Without a Stage-Wise Decoder

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2609.28327)

[Andrei Arhire](https://scholar.google.com/citations?user=BYkEZGFPq1wC&hl=en), [Mihaela Breaban](https://scholar.google.com/citations?user=i6CD3TIAAAAJ&hl=en), [Radu Timofte](https://scholar.google.com/citations?user=u3MwH5kAAAAJ&hl=en)


LightMIS employs a compact multi-scale encoder in which an Adaptive Fusion Cascade (AFC) processes features at every spatial scale. Each AFC combines AKF-based preprocessing, the proposed Progressive Receptive Fusion (PRF) module, and residual AKF refinement. PRF enriches narrow feature representations through temporary channel expansion, complementary depthwise receptive fields, and progressive cross-branch information transfer. For narrow-channel configurations, a conventional decoder introduces costly repeated upsampling and stage-wise encoder–decoder fusion. LightMIS therefore replaces it with a compact aggregation head: Scale-Aligned Projection (SAP) blocks process and align the output of every encoder level to a common resolution and channel width, after which the aligned features are fused and refined by a final AFC.

![LightMIS accuracy–latency Pareto front](assets/lightmis_accuracy_latency_pareto.png)

Evaluated under a common nnU-Net v2.3.1 protocol using five-fold cross-validation on six datasets spanning five imaging domains of 2D binary medical image segmentation, LightMIS achieves closely matched observed modality-macro performance on region-overlap metrics, together with competitive boundary-distance results relative to substantially larger models, including Mobile U-ViT, nnWNet, and nnU-Net. At the same time, LightMIS uses 90.58–99.61% fewer parameters and requires 82.54–96.14% fewer GFLOPs than these models. On a smartphone equipped with an Arm Mali-G52 MC2 GPU, all LightMIS variants achieve full GPU delegation (100% GPU operator coverage), with median delegated latency ranging from 53.31 ms for LightMIS-T to 138.31 ms for LightMIS.


## Contact

andrei.arhire@info.uaic.ro

