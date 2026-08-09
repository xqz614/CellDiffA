#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 DATASET VARIANT OUTPUT_DIR [GPU] [PERTURBDIFF_ROOT]"
  echo "  DATASET: pbmc | tahoe100m | replogle"
  echo "  VARIANT: scratch | finetuned"
}

if [[ $# -lt 3 || $# -gt 5 ]]; then
  usage >&2
  exit 2
fi

dataset="$1"
variant="$2"
output_dir="$3"
gpu="${4:-0}"
perturbdiff_root="${5:-external/PerturbDiff}"
data_root="${CELLDIFFA_DATA_ROOT:-/data/users/jchengak/DiffA/CellDiffA/data}"
perturb_data_root="$data_root/PerturbDiff_data"
checkpoint_root="$data_root/checkpoints/PerturbDiff_release_ckpt"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_input_dim=2000
pad_length=2000
embed_key="X_hvg"
variant_overrides=()
required_asset=""

case "$dataset" in
  pbmc)
    data_config="pbmc_finetune"
    checkpoint_dataset="pbmc"
    micro_batch=2048
    cell_set=256
    sample_flag="data.sample_pbmc_only=true"
    extra_covariates=("cov_encoding.celltype_encoding=llm")
    ;;
  tahoe100m)
    data_config="tahoe100m_finetune"
    checkpoint_dataset="tahoe100m"
    micro_batch=2048
    cell_set=256
    sample_flag="data.sample_tahoe100m_only=true"
    extra_covariates=("cov_encoding.celltype_encoding=llm")
    ;;
  replogle)
    data_config="replogle_finetune"
    checkpoint_dataset="replogle"
    micro_batch=128
    cell_set=32
    sample_flag="data.sample_replogle_only=true"
    extra_covariates=(
      "cov_encoding.celltype_encoding=llm"
      "cov_encoding.replogle_gene_encoding=genept"
    )
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

case "$variant" in
  scratch) checkpoint="$checkpoint_root/from_scratch_${checkpoint_dataset}.ckpt" ;;
  finetuned) checkpoint="$checkpoint_root/finetuned_${checkpoint_dataset}.ckpt" ;;
  *)
    usage >&2
    exit 2
    ;;
esac

# Replogle marginal-pretraining finetuning retains the 12,626-gene union used
# in pretraining. Its released checkpoint therefore cannot consume the 2,000
# HVG input used by the from-scratch checkpoint.
if [[ "$dataset" == "replogle" && "$variant" == "finetuned" ]]; then
  model_input_dim=12626
  pad_length=12626
  embed_key="X"
  merged_genes="$perturb_data_root/selected_genes/merged_pbmc_tahoe_rep_cellxgene_genes_mapped.pkl"
  required_asset="$merged_genes"
  variant_overrides=(
    "data.selected_gene_file=$merged_genes"
    "data.skip_cached_indices=true"
    "data.max_open_files=1000"
  )
fi

upstream_entrypoint="$perturbdiff_root/src/apps/run/rawdata_diffusion_sampling.py"
entrypoint="$script_dir/perturbdiff_sampling_entrypoint.py"
for required in "$upstream_entrypoint" "$entrypoint" "$checkpoint" "$perturb_data_root"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done
if [[ -n "$required_asset" && ! -e "$required_asset" ]]; then
  echo "Missing required path: $required_asset" >&2
  exit 1
fi

mkdir -p "$output_dir"
export CUDA_VISIBLE_DEVICES="$gpu"
export PERTURBDIFF_ROOT="$(cd "$perturbdiff_root" && pwd)"

# OmegaConf resolves relative interpolation from a list item differently from
# a mapping value. The upstream trixie_onehot config puts ${..path.tmp_dir}
# inside gene_embedding_path, which fails before sampling starts. Supply all
# covariate asset paths explicitly so the launcher is independent of that
# upstream interpolation bug.
gene_embedding_paths="[$perturb_data_root/gene_names/pbmc_highly_variavle_gene_emb_dict_emb_dict.pkl,$perturb_data_root/gene_names/replogle_gene_emb_dict_perturbation_emb_dict.pkl,$perturb_data_root/gene_names/replogle_highly_variavle_gene_emb_dict_emb_dict.pkl,$perturb_data_root/gene_names/tahoe100m_highly_variavle_gene_emb_dict_emb_dict.pkl]"

common=(
  "model_checkpoint_path=$checkpoint"
  # Select the upstream config groups explicitly before overriding their
  # fields. PerturbDiff's official sampling command does the same; without
  # these selections OmegaConf can fail while resolving ${path.tmp_dir}.
  "path=trixie_path"
  "cov_encoding=trixie_onehot"
  "data=$data_config"
  "data.normalize_counts=10"
  "data.num_workers=4"
  "data.prefetch_factor=16"
  "data.use_cell_set=$cell_set"
  "data.keep_control_cell=false"
  "optimization.micro_batch_size=$micro_batch"
  "model.hidden_num=[$model_input_dim,512]"
  "model.input_dim=$model_input_dim"
  "data.pad_length=$pad_length"
  "data.embed_key=$embed_key"
  "trainer.devices=[0]"
  "trainer.use_distributed_sampler=false"
  "device=cuda:0"
  "path.tmp_dir=$perturb_data_root"
  "path.diffusion.save_dir=$output_dir"
  "path.wandb.logging_dir=$output_dir/wandb"
  "sampling.output_dir=$output_dir"
  "sampling.num_sampled_batches=null"
  "sampling.use_ddim=true"
  "sampling.start_time=100"
  "sampling.eta=0.0"
  "sampling.guidance_strength=1.0"
  "cov_encoding.batch_encoding=onehot"
  "cov_encoding.gene_embedding_path=$gene_embedding_paths"
  "cov_encoding.pert_embedding_path=$perturb_data_root/meta_data/idx_to_pertemb.pkl"
  "cov_encoding.celltype_embedding_path=$perturb_data_root/meta_data/new_all_emb.pkl"
  "cov_encoding.drug_embedding_path=$perturb_data_root/meta_data/drug_embed_chemberta_cls_dict.pkl"
  "cov_encoding.replogle_gene_embedding_path=$perturb_data_root/gene_names/replogle_gene_emb_dict_perturbation_emb_dict.pkl"
  "model.p_drop_control=0"
  "$sample_flag"
  "lightning.logger._target_=pytorch_lightning.loggers.logger.DummyLogger"
  "~lightning.logger.project"
  "~lightning.logger.save_dir"
  "~lightning.logger.name"
)

if [[ "$dataset" == "replogle" && "$variant" == "finetuned" ]]; then
  python "$entrypoint" "${common[@]}" "${variant_overrides[@]}" "${extra_covariates[@]}"
else
  python "$entrypoint" "${common[@]}" "${extra_covariates[@]}"
fi
