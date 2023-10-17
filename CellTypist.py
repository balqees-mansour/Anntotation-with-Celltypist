#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:01:13 2023

@author: balqees
"""
''' Using CellTypist for cell type classification ''' 
# pip install celltypist
import scanpy as sc
import celltypist
from celltypist import models

anndata = sc.read("/home/balqees/Documents/convert seurat to anndata/processed_adata.h5ad")
anndata.shape
anndata.X
anndata.X.raw = anndata.X
anndata.X.expm1().sum(axis = 1)

## Assign cell type labels using a CellTypist built-in model

# Enabling `force_update = True` will overwrite existing (old) models.
models.download_models(force_update = True)

models.models_path

''' Get an overview of the models and what they represent.'''
models.models_description()

# Indeed, the `model` argument defaults to `Immune_All_Low.pkl`.
model = models.Model.load(model = 'Immune_All_Low.pkl')


(model.cell_types)

# Not run; predict cell identities using this loaded model.
#predictions = celltypist.annotate(adata_2000, model = model, majority_voting = True)
# Alternatively, just specify the model name (recommended as this ensures the model is intact every time it is loaded).
predictions = celltypist.annotate(anndata, model = 'Immune_All_Low.pkl', majority_voting = True, mode = 'best match')

predictions.predicted_labels

# Get an `AnnData` with predicted labels embedded into the cell metadata columns.
anndata = predictions.to_adata()
anndata.obs

# If the UMAP or any cell embeddings are already available in the `AnnData`, skip this command.
sc.tl.umap(anndata)



sc.pl.umap(anndata, color="HES4")
sc.pl.umap(anndata, color = ['predicted_labels'], size=60, legend_fontsize = "xx-small", save = "labelBM_low_umap.png")
sc.pl.umap(anndata, color = ['majority_voting'], size=60, legend_fontsize = "xx-small", save = "majorityBN_low_umap.png")

celltypist.dotplot(anndata, use_as_reference = 'predicted_labels', use_as_prediction = 'majority_voting')

sc.pl.umap(anndata, color = ['predicted_labels'], size=50,legend_loc = 'on data')
sc.pl.umap(
    anndata,
    color={'predicted_labels': {'size': 10}},  # Adjust 'size' to set the font size
    legend_loc='on data'
)

''' train on our custom model reference '''

adata_ref =  sc.read("/home/balqees/Downloads/adata_ref.h5ad")

adata_ref.var_names
adata_ref.obs_keys()

''' reference preprocessing '''
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

adata_ref.var_names_make_unique()  # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`
adata_ref

sc.pl.highest_expr_genes(adata_ref, n_top=20, )

sc.pp.filter_cells(adata_ref, min_genes=200)
sc.pp.filter_genes(adata_ref, min_cells=3)

adata_ref.var['mt'] = adata_ref.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata_ref, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

sc.pl.violin(adata_ref, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
            , jitter=0.4, multi_panel=True)

adata_ref = adata_ref[adata_ref.obs.n_genes_by_counts < 2500, :]
adata_ref = adata_ref[adata_ref.obs.pct_counts_mt < 5, :]

sc.pp.normalize_total(adata_ref, target_sum=1e4)

sc.pp.log1p(adata_ref)

sc.pp.highly_variable_genes(adata_ref, min_mean=0.0125, max_mean=3, min_disp=0.5)

sc.pl.highly_variable_genes(adata_ref)

adata_ref.raw = adata_ref
adata_ref = adata_ref[:, adata_ref.var.highly_variable]

sc.pp.regress_out(adata_ref, ['total_counts', 'pct_counts_mt'])

 #  sc.pp.scale(anndata, max_value=10)


# the `Cell_label` in `adata_ref.obs` will be used as cell type labels for training.
new_model = celltypist.train(adata_ref, labels = 'Cell_label', n_jobs = 10, feature_selection = True)

# Save the model.
new_model.write('/home/balqees/Documents/convert seurat to anndata/model_from_reference.pkl')

new_model = models.Model.load('/home/balqees/Documents/convert seurat to anndata/model_from_reference.pkl')

# Not run; predict the identity of each input cell with the new model.
#predictions = celltypist.annotate(adata_400, model = new_model, majority_voting = True)
# Alternatively, just specify the model path (recommended as this ensures the model is intact every time it is loaded).
predicts = celltypist.annotate(anndata, model = '/home/balqees/Documents/convert seurat to anndata/model_from_reference.pkl', majority_voting = True)

anndata = predicts.to_adata()

sc.tl.umap(anndata)

sc.pl.umap(anndata, color = ['predicted_labels'],size=40, legend_fontsize = "xx-small", save = "custom_model_umap.png")

sc.pl.umap(anndata, color = ['majority_voting'],size=40, legend_fontsize = "xx-small", save = "majority_umap.png")

anndata.obs_keys()





