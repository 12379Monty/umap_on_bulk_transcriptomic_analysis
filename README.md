# umap_on_bulk_transcriptomic_analysis
This code enables the projection of bulk-transcriptome profiles from one dataset into the UMAP embedding coordinates.


Requirements:
1) Python 3.6 or higher,
2) Numpy,
3) Scikit-learn,
4) Scipy,
5) Numba,
6) Seaborn,
7) UMAP (https://github.com/lmcinnes/umap),
8) openTSNE (https://github.com/pavlin-policar/openTSNE)

Run this code:
The input is the sample-gene matrix together with group labels, sledai_score, patient_id, visit_label, visit_date_label and day_from_start curated from GEO website
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE121239

Output: Figure 5a-e, Figure 6a,c
