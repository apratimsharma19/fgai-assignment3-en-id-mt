# English → Indonesian Machine Translation: Transformer vs mT5 vs Llama vs Gemma

FGAI (ICT4403) Assignment 3

## 1. Problem statement and objective
Build English→Indonesian machine-translation systems with four models (basic Transformer, mT5, Llama, Gemma), evaluate them on the same test set with BLEU, chrF, training/fine-tuning time, inference time and parameter count, compare encoder–decoder and decoder-only architectures, and discuss responsible-AI concerns.

## 2. Dataset
- **OPUS-100, en-id pair** — https://huggingface.co/datasets/Helsinki-NLP/opus-100
- Cleaning: dropped empty pairs, kept 3–50 words per side, dropped length ratio > 2.5, de-duplicated on English side, fixed-seed shuffle.
- Subset used: 50,000 train / 2,000 validation / 1,000 test pairs (LLM LoRA uses the first 10,000 train pairs).
- Statistics: see `outputs/dataset_stats.json`.  <!-- paste table after run -->

## 3. Models
| Model | Type | Adaptation |
|---|---|---|
| Basic Transformer (~20M) | encoder–decoder | from scratch, joint 8k SentencePiece BPE |
| mT5-small | encoder–decoder | full fine-tuning, fp32 |
| Llama-3.2-1B-Instruct (`unsloth/` mirror of the Meta weights) | decoder-only | 3-shot prompting + LoRA (r=16) |
| Gemma-3-1B-it (`unsloth/` mirror of the Google weights) | decoder-only | 3-shot prompting + LoRA (r=16) |

## 4. Execution steps and dependencies
1. Open `FGAI_A3_EN_ID_MT.ipynb` in Google Colab, select a **T4 GPU** runtime.
2. No Hugging Face login is needed (Llama and Gemma are loaded from ungated `unsloth/` mirrors).
3. Run all cells top to bottom. Outputs are written to `outputs/`.
4. Dependencies: `requirements.txt` (the first notebook cell installs them).

## 5. Results
<!-- paste outputs/results_table.md here after the run -->
Plots: `outputs/comparison.png`. Generated samples: `outputs/generated_samples.md`.

## 6. Model comparison and evaluation
<!-- Write from your real numbers. Cover: quality (BLEU/chrF), training cost, inference speed, parameters,
and the effect of pretraining, tokenization (8k joint BPE vs mT5 250k SentencePiece vs Llama/Gemma tokenizers)
and adaptation method (scratch / full FT / LoRA). State the caveat that LLMs saw 10k training pairs vs 50k. -->

## 7. Sampling-temperature analysis
<!-- paste outputs/temperature_table.md and temperature.png; temperatures 0.0, 0.5, 1.0 on the first 200 test sentences -->

## 8. Responsible and ethical AI
<!-- Base this on outputs/ethics_probe.md and outputs/flag_rates.json. Topics: gender bias (Indonesian "dia" is gender-neutral),
hallucinated numbers/entities, cultural misrepresentation (religion, festivals), idioms translated literally, sensitive health
or self-harm content. Measures: human review for high-stakes text, domain-specific evaluation, number/entity consistency checks,
confidence/uncertainty warnings, bias test sets, feedback channel. -->

## 9. Conclusion
<!-- Name the best-performing model and justify with the numbers. -->
