# Multilayer-Trajectory-Clustering
Trajectory Clustering for subtype identification in Parkinson's Disease

This code was used to produce the results in http://arxiv.org/abs/2005.14472


Abstract
Many diseases display heterogeneity in clinical features and their progression, indicative of the existence of disease subtypes. Extracting patterns of disease variable progression for subtypes has tremendous application in medicine, for example, in early prognosis and personalized medical therapy. This work present a novel, data-driven, network-based Trajectory Clustering (TC) algorithm for identifying Parkinson's subtypes based on disease trajectory. Modeling patient-variable interactions as a bipartite network, TC first extracts communities of co-expressing disease variables at different stages of progression. Then, it identifies Parkinson's subtypes by clustering similar patient trajectories that are characterized by severity of disease variables through a multi-layer network. Determination of trajectory similarity accounts for direct overlaps between trajectories as well as second-order similarities, i.e., common overlap with a third set of trajectories.
This work clusters trajectories across two types of layers: (a) temporal, and (b) ranges of independent outcome variable (representative of disease severity), both of which yield four distinct subtypes. The former subtypes exhibit differences in progression of disease domains (Cognitive, Mental Health etc.), whereas the latter subtypes exhibit different degrees of progression, i.e., some remain mild, whereas others show significant deterioration after 5 years.
The TC approach is validated through statistical analyses and consistency of the identified subtypes with medical literature. This generalizable and robust method can easily be extended to other progressive multi-variate disease datasets, and can effectively assist in targeted subtype-specific treatment in the field of personalized medicine.

trajectory_similarity.py clusters trajectories where the layers are temporal
along_UPDRS.py clusters trajectories where the layers are quartiles of the outcome variable (MDS-UPDRS-3)

Data can be obtained from Parkinson’s Progression Markers Initiative (PPMI) database (www.ppmi-info.org/data)
