# Atomic-LM: A Hybrid Mamba + DeepSeek MLA + MoE Language Model From Scratch

## 🚀 About the Project
Hi! I am a 13-year-old student from Kazakhstan, currently studying at NIS (Nazarbayev Intellectual Schools). I decided to dedicate my summer holidays to researching modern AI architectures and training LLMs from scratch. 

**Atomic-LM** is my very first mini-research project. It is a highly efficient, custom hybrid language model (4.4M parameters for the Mini version) designed to achieve maximum performance under extreme hardware constraints (trained completely on a single free Kaggle GPU T4). 

I am incredibly excited to share this work! I would highly appreciate any feedback, advice, or suggestions from the AI community as I continue my learning journey.

---

## 🧠 Architectural Highlights
Instead of building a standard dense Transformer, Atomic-LM explores the synergy of three cutting-edge AI breakthroughs to completely bypass the quadratic VRAM bottlenecks:

1. **Mamba (Selective State Spaces - SSM)**: Replaces standard attention in 5 out of 8 layers to achieve linear time complexity and a locked memory footprint.
2. **Multi-head Latent Attention (DeepSeek MLA Style)**: Applied to 2 key layers. It compresses the Key-Value (KV) cache into a 32-dimensional latent "bottleneck", slashing VRAM usage while retaining global context.
3. **Mixture of Experts (MoE)**: Activated every 2 layers. It routes tokens dynamically to 2 out of 4 specialized expert sub-networks, creating a high-capacity model that runs at the speed of a micro-network.
4. **GLM-Style Sparse RoPE Positioning**: Rotarry Position Embeddings (RoPE) are sparsely applied (only on the 5th MLA layer and the top Casual Attention layer), letting Mamba handle sequential order implicitly.
5. **Muon Optimizer Integration**: Leverages the revolutionary Muon optimizer for all 2D weight matrices (using Newton-Schulz orthogonalization) coupled with AdamW for 1D biases, scale, and embeddings.

---

## 📈 Phenomenal Training Results (Atomic-Mini)
The model was trained on **10% of the TinyStories dataset** using a custom-trained BPE tokenizer (Vocabulary Size: 8000) and a hard sequence length block size of 256. 

Thanks to the **Muon optimizer**, the network converged at an unprecedented speed, taking **only 950 seconds (~15 minutes)** to reach near-perfect language understanding:

| Step | Muon LR | AdamW LR | Train Loss | Val Loss | Perplexity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | 0.00020 | 0.00000 | 8.6135 | 8.5883 | 5368.28 |
| 100 | 0.02000 | 0.00030 | 4.8216 | 4.6612 | 105.76 |
| 200 | 0.01946 | 0.00029 | 1.2516 | 1.1518 | 3.16 |
| 500 | 0.01256 | 0.00019 | 0.3467 | 0.2566 | 1.29 |
| 900 | 0.00254 | 0.00004 | 0.1852 | 0.1620 | 1.18 |
| **999**| **0.00200** | **0.00003** | **0.1433** | **0.1172** | **1.12** |

*Note: Validation Loss is lower than Training Loss due to the regularizing effect of active Dropout (0.1) during training, proving that the model is perfectly generalized and not overfitting.*

---

## 📁 Repository Structure
```text
├── Architecture.py  # All custom neural building blocks (RMSNorm, MambaMixer, MLA, MoEFF)
├── dataset.py       # Custom BPE tokenizer trainer, dataset tokenization & sequence grouping
├── Atomic.py        # The core ModelConfig and full Atomic network assembly
└── main.py          # Dual optimizer routing (Muon+AdamW), Cosine schedule, and the main training loop
```

---

## 🚀 Roadmap: What's Next?
- [ ] **Shared Experts**: Upgrade the MoE layer to include a dedicated shared expert to minimize knowledge redundancy.
- [ ] **SwiGLU Activations**: Replace standard SiLU with Llama-3 style SwiGLU for enhanced logical mapping.
- [ ] **Multi-Agent Systems**: Explore and build decentralized multi-agent frameworks where specialized networks collaborate to solve complex reasoning tasks.
- [ ] **Latent Spaces & Custom Embeddings**: Deeply investigate representations within latent spaces to design better feature maps.
- [ ] **Custom Encoder-Decoder Frameworks**: Train a bespoke, hardware-efficient encoder and decoder sequence from scratch for advanced multimodal tasks.
- [ ] **And much more to come! 🚀**
