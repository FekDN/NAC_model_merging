# NAC_model_merging
Surgical Model Merging with the Neural Architecture Code (NAC) Genome

Traditional model merging techniques, such as weight averaging, SLERP, or TIES-merging, rely on a critical yet fragile assumption: **the models must have perfectly identical architectures.** This means the layer order, types, dimensions, and even the parameter names in the `state_dict` must match exactly.

This approach suffers from several major drawbacks:
*   **Brittleness:** The slightest architectural change (e.g., an extra `Dropout` layer) breaks the merging process entirely.
*   **Opacity:** You cannot be certain that two models are computationally identical without a manual code audit. Different implementations can produce the same parameter names but have subtle differences in their `forward` pass.
*   **Limited Scope:** Merging models from different architectural families (e.g., attempting to merge a `BERT` block with a `RoBERTa` block) is practically impossible.

The **Neural Architecture Code (NAC)** project transforms model merging from a "blind" process of averaging weights into an informed, surgical procedure based on a deep understanding of the model's computational "genome."

## Key Advantages of Merging with NAC

NAC provides a fundamentally more robust and powerful framework for model merging by addressing the core limitations of traditional methods.

### 1. Guaranteed Architectural Isomorphism
Before merging a single weight, NAC provides mathematical proof that the models are computationally identical.

*   **How it works:** By decomposing each model into its canonical NAC signature (a Base64 string), we can simply compare the strings. If they match, the computational graphs are guaranteed to be isomorphic. This is far more reliable than just comparing `state_dict` keys.
*   **Benefit:** Eliminates the risk of merging incompatible models, preventing silent errors and ensuring the resulting model is valid.

### 2. Granular, Block-Level Merging ("Knowledge Transplantation")
NAC identifies semantically equivalent functional blocks ("genes") across models, even if their implementations or parameter names differ.

*   **How it works:** The NAC `registry.json` contains a library of discovered patterns (e.g., `TransformerBlock`, `ResidualConnection`). We can identify these patterns within different models and merge them selectively.
*   **Benefit:** This enables powerful new capabilities, such as transplanting the encoder from a model trained on medical texts with the classification head of a model trained on general knowledge. You merge *skills*, not just entire models.

### 3. Intelligent and Automatic Parameter Mapping
NAC abstracts away from fragile, implementation-specific parameter names (e.g., `bert.encoder.layer.0...`) to stable, canonical integer IDs.

*   **How it works:** If two models are architecturally identical, a specific parameter (like the query projection matrix in the first attention block) will have the *exact same parameter ID* in the NAC registry, regardless of its string name in the `state_dict`.
*   **Benefit:** The process of matching weights between models becomes fully automatic and error-proof. No more complex and brittle regex rules to align parameter names.

### 4. Enables Future Cross-Family Merging
Because NAC operates on a fundamental computational level (the ATen instruction set), it can identify semantic similarities between blocks that are not obvious at the source code level.

*   **How it works:** A `Conv-BN-ReLU` block from a ResNet and a `DepthwiseConv-LayerNorm-GELU` block from a ConvNeXt may share a sub-sequence of fundamental NAC operations.
*   **Benefit:** This opens the door to merging functional blocks from different model families, creating novel hybrid architectures by combining the best "genes" from a diverse pool of models.

---

## The NAC Merging Algorithm

Merging two models `A` and `B` with NAC is a structured, verifiable process:

1.  **Decomposition & Sequencing:** Both `model_A` and `model_B` are fed into the `Decomposer`. This traces their computational graphs and converts them into canonical NAC binary representations.

2.  **Architectural Verification:** The Base64 NAC signatures of the two models (or the specific parts to be merged, like the `base_model`) are compared.
    *   **If `signature_A == signature_B`**: The models are architecturally identical. Proceed to the next step.
    *   **If `signature_A != signature_B`**: The models are incompatible. Abort the merge with a clear error.

3.  **Canonical Parameter Mapping:** Using the shared `OperationRegistry`, we iterate through the universal list of parameter names (`index_to_param_name`). For each canonical parameter name, we retrieve the corresponding tensors from `model_A.state_dict()` and `model_B.state_dict()`.

4.  **Weight Fusion:** With the parameters correctly aligned, the desired merging strategy (e.g., `average`, `SLERP`, `TIES`) is applied to each pair of tensors to produce the new, merged tensor.

5.  **Model Instantiation:** A new model instance is created, and the newly created `merged_state_dict` is loaded into it.

---

## Code Example

The following Python function demonstrates a conceptual implementation of the NAC-powered model merging workflow.

```python
import torch
import torch.nn as nn
from collections import OrderedDict

# Assume all NAC classes (Decomposer, OperationRegistry, etc.) are defined or imported.
# from nac_compiler import Decomposer, OperationRegistry 

def merge_models_with_nac(
    model_A: nn.Module,
    model_B: nn.Module,
    model_meta_A: dict,
    model_meta_B: dict,
    registry: 'OperationRegistry',
    decomposer: 'Decomposer',
    merge_strategy: str = 'average'
) -> nn.Module:
    """
    Merges two models using NAC to verify compatibility and map parameters intelligently.

    Args:
        model_A: The first model to merge.
        model_B: The second model to merge.
        model_meta_A: Metadata required by the Decomposer for model A.
        model_meta_B: Metadata required by the Decomposer for model B.
        registry: A loaded OperationRegistry containing the shared vocabulary.
        decomposer: An initialized Decomposer instance.
        merge_strategy: The fusion strategy to apply ('average', 'slerp', etc.).

    Returns:
        A new, merged nn.Module, or None if the models are architecturally incompatible.
    """
    print("--- Starting Model Merge with NAC Verification ---")

    # Step 1: Decompose both models into their NAC graph representations.
    # We focus on the 'base_model' as heads can legitimately differ.
    print("Step 1: Decomposing models into NAC representations...")
    graphs_A = decomposer.run(model_A, "model_A", model_meta_A)
    graphs_B = decomposer.run(model_B, "model_B", model_meta_B)

    base_model_graph_A = graphs_A.get('base_model')
    base_model_graph_B = graphs_B.get('base_model')

    if not base_model_graph_A or not base_model_graph_B:
        print("Error: Could not extract 'base_model' from one or both models.")
        return None

    # Step 2: Verify architectural isomorphism by comparing NAC signatures.
    print("Step 2: Verifying architectural compatibility...")
    signature_A = base_model_graph_A.to_base64()
    signature_B = base_model_graph_B.to_base64()

    if signature_A != signature_B:
        print("!!!!! CRITICAL ERROR: Models are architecturally incompatible. Merge aborted.")
        print(f"  Signature A length: {len(signature_A)}")
        print(f"  Signature B length: {len(signature_B)}")
        return None
    
    print("  -> Success! Models are fully compatible at the computational graph level.")

    # Step 3: Instantiate a new model (as a copy of A) to host the merged weights.
    print("Step 3: Creating new model and merging weights...")
    
    # In a real application, you might need a more robust way to instantiate the model.
    merged_model = type(model_A)(model_A.config).eval()
    
    state_dict_A = model_A.state_dict()
    state_dict_B = model_B.state_dict()
    merged_state_dict = OrderedDict()

    # Step 4: Perform intelligent parameter mapping via the NAC Registry.
    print("Step 4: Mapping parameters using the canonical NAC registry...")
    
    # The registry's index_to_param_name provides the canonical list of all parameters.
    # The integer keys are the universal IDs shared by both compatible models.
    num_params_merged = 0
    for param_id, param_name in registry.index_to_param_name.items():
        if param_name in state_dict_A and param_name in state_dict_B:
            tensor_A = state_dict_A[param_name]
            tensor_B = state_dict_B[param_name]

            if tensor_A.shape != tensor_B.shape:
                print(f"  -> Warning: Skipping param '{param_name}' due to shape mismatch.")
                merged_state_dict[param_name] = tensor_A # Default to model A's tensor
                continue

            # Step 5: Apply the fusion strategy.
            if merge_strategy == 'average':
                merged_tensor = (tensor_A + tensor_B) / 2
            # Add other strategies like SLERP here.
            # elif merge_strategy == 'slerp':
            #     ...
            else:
                raise ValueError(f"Unknown merge strategy: {merge_strategy}")

            merged_state_dict[param_name] = merged_tensor
            num_params_merged += 1
        elif param_name in state_dict_A:
            # If a parameter only exists in one model (e.g., a classification head), copy it.
             merged_state_dict[param_name] = state_dict_A[param_name]

    print(f"  -> Successfully merged {num_params_merged} shared parameters.")
    
    # Load the new state_dict into the model.
    merged_model.load_state_dict(merged_state_dict, strict=False)
    
    print("--- Model merging complete! ---")
    return merged_model

```

---

## Synergy with Advanced Architectures: NAC and EMM

While NAC provides a powerful framework for merging structurally identical models, advanced research systems like the **Elastic Memory Model (EMM)** tackle the challenge of merging *heterogeneous* architectures.

*   **EMM ([https://github.com/FekDN/EMM](https://github.com/FekDN/EMM))** is a system that assimilates knowledge from diverse expert models into a single, consolidated network. It operates by analyzing the *functional similarity* of layers (how they behave on given inputs) rather than their structure. To do this, it uses complex, heuristic-based metrics like Centered Kernel Alignment (CKA) and Singular Vector Canonical Correlation Analysis (SVCCA) to estimate if two different layers perform a similar role.

The two approaches represent two sides of the same coin: **structural analysis vs. functional analysis**.

| Approach | NAC (Structural "Genomics") | EMM (Functional "Neurophysiology") |
| :--- | :--- | :--- |
| **Method** | Compares the fundamental binary code (`ABCD[]`) of model components. | Measures and compares activation patterns from live model inference. |
| **Goal** | Find **architecturally identical** components ("genes"). | Find **functionally similar** components ("cognitive roles"). |
| **Reliability** | **Deterministic & Mathematical.** A match is a 100% guarantee of isomorphism. | **Heuristic & Probabilistic.** A match is a high-probability estimate based on complex metrics and thresholds. |
| **Scope** | Merging identical or near-identical models and blocks. | Merging completely different architectures (e.g., BERT + RoBERTa). |

**The true potential lies in their integration.** NAC allows merging models and their parts (at the "gene" or pattern level) at a more fundamental and reliable level than the current implementation in EMM. If the NAC standard were adopted, its integration into EMM would revolutionize the merging process by **replacing complex heuristics with mathematical precision.**

Instead of relying solely on functional similarity, an NAC-powered EMM would first check for structural identity using NAC signatures. This would make the assimilation process faster, more reliable, and capable of even more granular, sub-layer merging, truly unlocking the potential of a universal AI knowledge base.

---

## Learn More About NAC

This merging capability is just one application of the NAC framework. To explore the full project, including the specification, compiler, and the vision for a universal AI genome, please visit the main repository:

**[https://github.com/FekDN/NAC](https://github.com/FekDN/NAC)**

---

*   feklindn@gmail.com 

---

## License

The source code of this project is licensed under the **GNU General Public License v3.0**. See the `LICENSE` file for details.

The accompanying documentation, including this README and the project's White Paper, is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License**.
