Final Implementation Plan: Redesigning Tokenization for Fragment Detection and Assembly
The goal is to redesign the tokenization pipeline (currently in tokenize_swc_vq.py) to move away from a strict, complete-tree Depth-First Search (DFS) grammar towards a flexible "fragment-based" grammar. This new approach will be suitable for detecting, assembling, and joining arbitrary neuron fragments from raw volumes or noisy images, while preserving the mathematical graph topology of the neuron.

1. Token Vocabulary Design
To satisfy the requirements of scale invariance, manageable vocabulary size, and reversibility, we will use a 3-stream token tuple for every step along a fragment: [<REL_LOC_X>, <DIR_X>, <MORPH_X>] plus a region token <REG_X>.

A. Relative Location (<REL_LOC_X>)
Concept: Instead of absolute CCFv3 coordinates, location steps will be encoded as relative spatial vectors $\Delta \vec{v} = (\Delta X, \Delta Y, \Delta Z)$ between connected nodes.
Scale Invariance & CCFv3: To ensure scale invariance across any 3D space, the relative vectors will be normalized by a local scale factor (e.g., unit vectors or normalized by median segment length). The mapping to CCFv3 space will be anchored by a coarse <REG_X> token at the start of a fragment.
Manageable & Reversible: We will use K-Means clustering on these relative vectors to create a finite, manageable vocabulary (e.g., 512 clusters). This is perfectly reversible during detokenization by accumulating the relative vector cluster centers.
B. Direction / Tracing (<DIR_X>)
Concept: To aid in spatial tracing through noisy volumes, we will explicitly cluster the absolute unit tangent direction of the segment.
Vocabulary: K-Means clustering on the 3D unit sphere.
C. Invariant Morphology (<MORPH_X>)
Concept: Purely scale and rotation-invariant features.
Features: Curvature (angle change between the current and previous relative vector), torsion, and radius/thickness ratio.
Vocabulary: K-Means clustering on these invariant geometric properties.
2. Fragment-Based Topological Grammar
To support incomplete fragments while strictly adhering to the graph structure and topology of the neuron, we will decompose the SWC graph into a list of unbranched Linear Segments (Fragments).

Instead of deep nested brackets (<BIF> ... <POP>), each fragment will be serialized sequentially with explicit topological boundaries:

Fragment Start Tokens:

<START_SOMA>: The root of a complete neuron.
<START_FRAG>: A disconnected or incomplete fragment starting point in space.
<START_BIF>: A fragment that originates from a bifurcation point of a previously emitted fragment.
Fragment End Tokens:

<END_LEAF>: The fragment naturally terminates at a dead-end tip.
<END_BIF>: The fragment bifurcates (splits). The model knows to expect multiple subsequent <START_BIF> fragments originating from this location.
Example Sequence:

text

<START_SOMA> [Tokens...] <END_BIF>
<START_BIF>  [Tokens...] <END_LEAF>
<START_BIF>  [Tokens...] <END_LEAF>
<START_FRAG> [Tokens...] <END_LEAF>   # A disconnected fragment detected nearby
Proposed Changes & File Modifications
1. tokenize_fragments.py [NEW]
A completely new script to replace the DFS tokenizer. It will:

Parse SWCs into a directed graph.
Decompose the graph into unbranched linear fragments.
Compute the relative locations, tangents, and morphological features.
Train the 3 new K-Means vocabularies (kmeans_rel_loc.pkl, kmeans_dir.pkl, kmeans_morph.pkl).
Output a dataset where each JSON line contains the serialized fragment sequences.
2. detokenize_fragments.py [NEW]
A new script to convert the generated token sequences back into SWC files. It will:

Reconstruct fragments by accumulating <REL_LOC_X> cluster centers.
Reconnect fragments topologically when it sees <END_BIF> followed by <START_BIF>.
Handle disconnected <START_FRAG> components gracefully.
Verification Plan
Automated Tests
Run python tokenize_fragments.py on a small subset of SWCs to ensure vocabularies train correctly.
Run python detokenize_fragments.py on the generated sequences.
Compare reconstruction accuracy (Total Length, Node Count) of the fragment-based approach compared to the original SWCs to guarantee mathematical reversibility.