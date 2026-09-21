#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 VARIANT OUTPUT_DIR [GPU] [WORKER_INDEX] [NUM_WORKERS] [MAX_GROUPS] [PERTURBDIFF_ROOT]"
  echo "  VARIANT: scratch | finetuned"
  echo "  MAX_GROUPS: omit or use 'all' for a complete worker run"
}

if [[ $# -lt 2 || $# -gt 7 ]]; then
  usage >&2
  exit 2
fi

variant="$1"
output_dir="$2"
gpu="${3:-0}"
device="${CELLDIFFA_DEVICE:-cuda:0}"
runtime_overrides=("device=$device")
if [[ "$device" == "mps" || "$device" == "cpu" ]]; then
  runtime_overrides=(
    "trainer.accelerator=$device" "trainer.devices=1"
    "data.num_workers=0" "data.prefetch_factor=null"
    "data.persistent_workers=false" "data.pin_memory=false"
  )
fi
worker_index="${4:-0}"
num_workers="${5:-1}"
max_groups="${6:-all}"
perturbdiff_root="${7:-external/PerturbDiff}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
if [[ "$perturbdiff_root" = /* ]]; then
  perturbdiff_root_abs="$(cd "$perturbdiff_root" && pwd)"
else
  perturbdiff_root_abs="$(cd "$repo_root/$perturbdiff_root" && pwd)"
fi
data_root="${CELLDIFFA_DATA_ROOT:-$repo_root/data}"
perturb_data_root="$data_root/PerturbDiff_data"
checkpoint_root="$data_root/checkpoints/PerturbDiff_release_ckpt"
source_h5ad="$perturb_data_root/finetune_data/nadig_processed_data/replogle.h5ad"
selected_genes="$perturb_data_root/selected_genes/replogle_real_selected_genes.pkl"
perturbation_embeddings="$perturb_data_root/gene_names/replogle_gene_emb_dict_perturbation_emb_dict.pkl"
evaluation_split="${CELLDIFFA_EVALUATION_SPLIT:-test}"
reference_filename="real.h5ad"
if [[ "$evaluation_split" == "validation" ]]; then
  reference_filename="validation.h5ad"
fi
real_test="${CELLDIFFA_REAL_TEST:-$repo_root/results/replogle/reference/$reference_filename}"
split_config="$perturbdiff_root_abs/configs/data/perturb_data/replogle.yaml"
entrypoint="$repo_root/scripts/baselines/run_celldiffa_replogle.py"
model_input_dim=2000
embed_key="X_hvg"
pad_length=2000
variant_overrides=()

case "$variant" in
  scratch)
    checkpoint="$checkpoint_root/from_scratch_replogle.ckpt"
    ;;
  finetuned)
    checkpoint="$checkpoint_root/finetuned_replogle.ckpt"
    model_input_dim=12626
    embed_key="X"
    pad_length=12626
    merged_genes="$perturb_data_root/selected_genes/merged_pbmc_tahoe_rep_cellxgene_genes_mapped.pkl"
    variant_overrides=(
      "data.selected_gene_file=$merged_genes"
      "data.skip_cached_indices=true"
      "data.max_open_files=1000"
    )
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

for path in \
  "$entrypoint" \
  "$perturbdiff_root_abs/src/apps/run/rawdata_diffusion_sampling.py" \
  "$source_h5ad" \
  "$selected_genes" \
  "$perturbation_embeddings" \
  "$real_test" \
  "$split_config" \
  "$checkpoint"
do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 1
  fi
done
if [[ "$variant" == "finetuned" && ! -e "$merged_genes" ]]; then
  echo "Missing required input: $merged_genes" >&2
  exit 1
fi

mkdir -p "$output_dir/shards"
worker_output_dir="$output_dir/workers/worker_${worker_index}"
mkdir -p "$worker_output_dir"
export CUDA_VISIBLE_DEVICES="$gpu"
export PERTURBDIFF_ROOT="$perturbdiff_root_abs"
export PYTHONPATH="$repo_root:${PYTHONPATH:-}"

gene_embedding_paths="[$perturb_data_root/gene_names/pbmc_highly_variavle_gene_emb_dict_emb_dict.pkl,$perturb_data_root/gene_names/replogle_gene_emb_dict_perturbation_emb_dict.pkl,$perturb_data_root/gene_names/replogle_highly_variavle_gene_emb_dict_emb_dict.pkl,$perturb_data_root/gene_names/tahoe100m_highly_variavle_gene_emb_dict_emb_dict.pkl]"

celldiffa_args=(
  "--source" "$source_h5ad"
  "--real-test" "$real_test"
  "--split-config" "$split_config"
  "--selected-genes" "$selected_genes"
  "--perturbation-embeddings" "$perturbation_embeddings"
  "--prior-cache" "$output_dir/training_priors.npz"
  "--shard-root" "$output_dir/shards"
  "--output" "$output_dir/celldiffa_${variant}.h5ad"
  "--variant" "$variant"
  "--evaluation-split" "$evaluation_split"
  "--alpha" "${CELLDIFFA_ALPHA:-1.0}"
  "--alignment-mode" "${CELLDIFFA_ALIGNMENT_MODE:-smc}"
  "--reward-unit" "${CELLDIFFA_REWARD_UNIT:-population}"
  "--reward-normalization" "${CELLDIFFA_REWARD_NORMALIZATION:-zscore}"
  "--ess-threshold" "${CELLDIFFA_ESS_THRESHOLD:-0.5}"
  "--seed" "${CELLDIFFA_SEED:-42}"
  "--prior-ridge" "${CELLDIFFA_PRIOR_RIDGE:-1.0}"
  "--anchor-bandwidth" "${CELLDIFFA_ANCHOR_BANDWIDTH:-1.0}"
  "--reward-weights" "${CELLDIFFA_SIGNATURE_WEIGHT:-1.0}" "${CELLDIFFA_DIRECTION_WEIGHT:-1.0}" "${CELLDIFFA_ANCHOR_WEIGHT:-1.0}"
  "--worker-index" "$worker_index"
  "--num-workers" "$num_workers"
  "--num-particles" "${CELLDIFFA_NUM_PARTICLES:-16}"
  "--particle-batch-cells" "${CELLDIFFA_PARTICLE_BATCH_CELLS:-128}"
  "--native-blocks-per-population" "${CELLDIFFA_NATIVE_BLOCKS_PER_POPULATION:-1}"
)
if [[ "$max_groups" != "all" ]]; then
  celldiffa_args+=("--max-groups" "$max_groups")
fi

hydra_args=(
  "model_checkpoint_path=$checkpoint"
  "path=trixie_path"
  "cov_encoding=trixie_onehot"
  "data=replogle_finetune"
  "data.normalize_counts=10"
  "data.num_workers=4"
  "data.prefetch_factor=16"
  "data.use_cell_set=32"
  "data.keep_control_cell=false"
  "optimization.micro_batch_size=128"
  "model.hidden_num=[$model_input_dim,512]"
  "model.input_dim=$model_input_dim"
  "data.pad_length=$pad_length"
  "data.embed_key=$embed_key"
  "trainer.devices=[0]"
  "trainer.use_distributed_sampler=false"
  "device=$device"
  "path.tmp_dir=$perturb_data_root"
  "path.diffusion.save_dir=$worker_output_dir"
  "path.wandb.logging_dir=$worker_output_dir/wandb"
  "sampling.output_dir=$worker_output_dir"
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
  "cov_encoding.celltype_encoding=llm"
  "cov_encoding.replogle_gene_encoding=genept"
  "model.p_drop_control=0"
  "data.sample_replogle_only=true"
  "lightning.logger._target_=pytorch_lightning.loggers.logger.DummyLogger"
  "~lightning.logger.project"
  "~lightning.logger.save_dir"
  "~lightning.logger.name"
)

if [[ "$variant" == "finetuned" ]]; then
  python "$entrypoint" \
    "${celldiffa_args[@]}" \
    "${hydra_args[@]}" \
    "${variant_overrides[@]}" \
    "${runtime_overrides[@]}"
else
  python "$entrypoint" \
    "${celldiffa_args[@]}" \
    "${hydra_args[@]}" \
    "${runtime_overrides[@]}"
fi
