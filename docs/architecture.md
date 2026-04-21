# M3-Agent: Architecture & Module Reference

This document provides a detailed explanation of the M3-Agent modeling architecture and modules. It covers the two main pipelines (memorization and control), the core data structure (VideoGraph), and every module involved in the system.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [VideoGraph: The Core Data Structure](#videograph-the-core-data-structure)
3. [Memorization Pipeline](#memorization-pipeline)
   - [Video Preprocessing](#video-preprocessing)
   - [Face Processing](#face-processing)
   - [Voice Processing](#voice-processing)
   - [Memory Generation](#memory-generation)
   - [Memory Processing & Graph Construction](#memory-processing--graph-construction)
4. [Control Pipeline](#control-pipeline)
   - [Multi-Turn Reasoning Loop](#multi-turn-reasoning-loop)
   - [Retrieval Module](#retrieval-module)
   - [Answer Evaluation](#answer-evaluation)
5. [Prompt System](#prompt-system)
6. [API & Inference Backends](#api--inference-backends)
7. [Character Identity Resolution](#character-identity-resolution)

---

## System Overview

M3-Agent operates in two parallel phases:

```
                        ┌─────────────────────────────────────┐
                        │         MEMORIZATION (offline)       │
                        │                                     │
  Video ──► Clip ──►    │  Face Detection ──┐                 │
  (30s segments)        │  Voice Diarize ───┼──► LLM ──► VideoGraph (pickle)
                        │  Frame Extraction ┘                 │
                        └─────────────────────────────────────┘

                        ┌─────────────────────────────────────┐
                        │           CONTROL (online)          │
                        │                                     │
  Question ──►          │  Think ──► Search ──► Retrieve ──►  │
                        │     ▲         │         │           │
                        │     └─────────┘    VideoGraph       │  ──► Answer
                        │  (multi-turn loop, up to N rounds)  │
                        └─────────────────────────────────────┘
```

**Memorization** processes video clips sequentially, extracting faces and voices as perceptual anchors, then uses a multimodal LLM (Qwen2.5-Omni or GPT-4o/Gemini) to generate episodic and semantic memory sentences. These are embedded and inserted into a `VideoGraph`.

**Control** takes a question and iteratively reasons over the VideoGraph. At each step, the model either produces a final answer or generates a search query. The query is embedded, matched against graph nodes via cosine similarity, and the retrieved memories are fed back into the next reasoning turn.

---

## VideoGraph: The Core Data Structure

**File:** `mmagent/videograph.py`

VideoGraph is a multimodal graph that serves as the agent's long-term memory. It is serialized as a pickle file and loaded at inference time.

### Node Types

| Type | Stored In | Embeddings | Metadata |
|------|-----------|------------|----------|
| `img` (face) | `nodes` dict | ArcFace face embeddings (up to `max_img_embeddings`) | `contents`: list of base64-encoded cropped face images |
| `voice` | `nodes` dict | ERes2NetV2 speaker embeddings (up to `max_audio_embeddings`) | `contents`: list of ASR transcript strings |
| `episodic` | `nodes` dict + `text_nodes` list | OpenAI `text-embedding-3-large` vectors | `contents`: list of atomic event description strings; `timestamp`: clip ID |
| `semantic` | `nodes` dict + `text_nodes` list | OpenAI `text-embedding-3-large` vectors | `contents`: list of high-level conclusion strings; `timestamp`: clip ID |

### Edges

Edges are bidirectional weighted connections stored in `edges` dict keyed by `(node_id1, node_id2)`. Rules:
- Text nodes connect to the face/voice entity nodes they mention (parsed via `<face_X>` / `<voice_X>` tags).
- Same-type text-to-text edges are not allowed (no episodic-episodic or semantic-semantic edges).
- Edge weights can be reinforced (+1) or weakened (-1). Edges with weight <= 0 are removed.

### Key Indexes

- `text_nodes`: ordered list of all text node IDs (insertion order).
- `text_nodes_by_clip`: `{clip_id: [node_ids]}` — lookup text nodes by video clip.
- `event_sequence_by_clip`: `{clip_id: [episodic_node_ids]}` — episodic events per clip.
- `character_mappings`: `{character_X: [face_Y, voice_Z, ...]}` — unified character identities.
- `reverse_character_mappings`: `{face_Y: character_X}` — reverse lookup.

### Search Methods

- **`search_text_nodes(query_embeddings, range_nodes, mode)`**: Vectorized cosine similarity search over all text nodes (or a restricted subset connected to `range_nodes`). Modes: `max`, `mean`, `sum`, `min` across query-node embedding pairs. Returns `[(node_id, score)]` sorted descending.
- **`search_img_nodes(img_info)`**: Finds face nodes whose average cosine similarity to the query face embeddings exceeds `img_matching_threshold` (default 0.3).
- **`search_voice_nodes(audio_info)`**: Same as above for voice nodes with `audio_matching_threshold` (default 0.6).

### Graph Maintenance

- **`refresh_equivalences()`**: Uses a Union-Find (disjoint set) algorithm to discover which face and voice nodes refer to the same character, based on "Equivalence: <face_X>, <voice_Y>" semantic nodes. Builds `character_mappings` and `reverse_character_mappings`. Also resolves collision among competing equivalence claims via `fix_collisions()`.
- **`fix_collisions(node_id, mode)`**: For a given voice node, resolves conflicting face-voice equivalence claims. In `eq_only` mode, keeps only the equivalence node with the highest edge weight. In `argmax`/`dropout` modes, uses DBSCAN clustering on semantic node embeddings to detect redundant semantic memories, then keeps the strongest or samples by weight.
- **`_cluster_semantic_nodes(nodes, threshold)`**: Clusters semantic nodes using DBSCAN on a cosine-distance matrix (distance = 1 - similarity). Nodes within `eps = 1 - threshold` are grouped.
- **`truncate_memory_by_clip(clip_id)`**: Removes all nodes/edges with IDs after the last node of the given clip. Used at inference to simulate memory up to a specific time point.
- **`prune_memory_by_node_type(node_type)`**: Removes all nodes of a given type (e.g., all semantic nodes for ablation studies).

### Training Support

- **`expand_route(route)` / `sample_a_route(length)`**: Samples random paths through the graph by following entity connections. Used for generating training data — walks from a random text node, picks a mentioned entity, then jumps to another text node connected to that entity.

---

## Memorization Pipeline

Entry point: `m3_agent/memorization_memory_graphs.py`

For each video, clips are processed sequentially. Each clip goes through four stages:

### Video Preprocessing

**File:** `mmagent/utils/video_processing.py`

`process_video_clip(video_path, fps=5, audio_fps=16000)` extracts three representations from a 30-second clip:
- **`base64_video`**: The raw video file bytes, base64-encoded (sent to Gemini/Qwen for visual understanding).
- **`base64_frames`**: Individual frames sampled at `fps` (default 5 FPS = ~150 frames per 30s clip), each JPEG-encoded and base64-encoded. Used for face detection.
- **`base64_audio`**: Audio extracted to WAV at 16kHz, base64-encoded. Used for speaker embedding.

### Face Processing

**File:** `mmagent/face_processing.py`  
**Low-level:** `mmagent/src/face_extraction.py`, `mmagent/src/face_clustering.py`

**Step 1: Face Detection & Embedding** (`extract_faces`)  
Uses InsightFace's `buffalo_l` model (RetinaFace detector + ArcFace recognizer). For each frame:
- Detects all faces, extracts bounding boxes.
- Computes 512-dim normalized ArcFace embeddings.
- Computes detection score (confidence) and quality score (L2 norm of raw embedding).
- Crops face region and base64-encodes it.
- Classifies face as "ortho" (frontal, aspect ratio 1.0-1.5) or "side".

Frames are processed in parallel batches using `ThreadPoolExecutor`.

**Step 2: Intra-Clip Clustering** (`cluster_faces`)  
Uses HDBSCAN on cosine distance matrix of face embeddings:
- Only "good" faces (detection score >= 0.8, quality score >= 20) participate in clustering.
- Bad-quality faces are left unclustered (cluster_id = -1).
- `min_cluster_size=20`, `distance_threshold=0.5`.

**Step 3: Graph Update** (`update_videograph`)  
For each cluster of faces in the current clip:
- Search existing img nodes in the VideoGraph via `search_img_nodes`.
- If a match is found (similarity >= `img_matching_threshold`), update that node's embeddings.
- Otherwise, create a new img node.
- Each node stores up to `max_img_embeddings` embeddings; if exceeded, randomly samples down.

**Quality Filtering:** Only faces with detection score > `face_detection_score_threshold` (0.85) and quality score > `face_quality_score_threshold` (22) are kept. Top `max_faces_per_character` (3) faces per cluster are retained, sorted by detection and quality score.

**Caching:** Intermediate results are saved to JSON files (`clip_X_faces.json`). If the file exists, detection is skipped.

### Voice Processing

**File:** `mmagent/voice_processing.py`

**Step 1: Audio Diarization** (`diarize_audio`)  
Sends the base64-encoded video to Gemini-1.5-Pro with `prompt_audio_segmentation`. The model performs ASR and speaker turn segmentation, returning JSON with `{start_time, end_time, asr}` for each speech segment. Segments shorter than `min_duration_for_audio` (2 seconds) are filtered out.

**Step 2: Audio Segment Extraction** (`create_audio_segments`)  
Decodes the base64 audio (WAV format), slices it at the timestamps from diarization using pydub, and re-encodes each segment as base64 WAV.

**Step 3: Speaker Embedding** (`get_normed_audio_embeddings`)  
Each audio segment is processed by **ERes2NetV2** (loaded from `models/pretrained_eres2netv2.ckpt`):
- FBank features extracted at 80-dim, 16kHz.
- Model outputs 192-dim speaker embedding.
- Embeddings are L2-normalized.

**Step 4: Graph Update** (`update_videograph`)  
Same pattern as face processing:
- Search existing voice nodes via `search_voice_nodes`.
- If match found (similarity >= `audio_matching_threshold`, default 0.6), update the node.
- Otherwise, create a new voice node.

**Caching:** Results saved to `clip_X_voices.json`.

### Memory Generation

Two backends produce episodic and semantic memory sentences from the video context:

#### GPT-4o / Gemini Backend

**File:** `mmagent/memory_processing.py`

`generate_video_context()` assembles a multimodal prompt:
- The base64 video clip.
- Face images: for each detected face cluster, the best face crop with its `<face_X>` ID label, drawn on the original frame with a green bounding box.
- Voice segments: for each voice node, the ASR transcripts with timestamps, labeled as `<voice_X>`.

`generate_full_memories()` sends this context + `prompt_generate_full_memory` to Gemini-1.5-Pro. The model returns a Python dict with two keys:
- `episodic_memory`: list of atomic event descriptions referencing `<face_X>` and `<voice_X>` IDs.
- `semantic_memory`: list of high-level conclusions including equivalence declarations ("Equivalence: <face_X>, <voice_Y>"), character attributes, relationships, plot understanding, and general knowledge.

#### Qwen2.5-Omni Backend

**File:** `mmagent/memory_processing_qwen.py`

Uses the fine-tuned **M3-Agent-Memorization** model (based on Qwen2.5-Omni-7B). The model is loaded locally via HuggingFace Transformers with Flash Attention 2.

`generate_memories()` assembles a similar multimodal prompt but formatted for the Qwen processor (video path, face images, voice transcripts). Uses `prompt_generate_memory_with_ids_sft` which requests a JSON output with `video_description` (episodic) and `high_level_conclusions` (semantic).

The Qwen backend enables audio-in-video processing (`USE_AUDIO_IN_VIDEO = True`), so the model can jointly reason over visual and auditory signals.

### Memory Processing & Graph Construction

**Files:** `mmagent/memory_processing.py` and `mmagent/memory_processing_qwen.py` (both contain `process_memories()`)

After memory sentences are generated, they are embedded and inserted into the VideoGraph:

**Step 1: Embedding**  
Each memory sentence is embedded using OpenAI `text-embedding-3-large` via `parallel_get_embedding`.

**Step 2: Episodic Memory Insertion**  
Each episodic memory creates a new text node. Entity references (`<face_X>`, `<voice_X>`) are parsed, and edges are added connecting the text node to the referenced entity nodes.

**Step 3: Semantic Memory Insertion (with deduplication)**  
Semantic memories undergo a more sophisticated insertion process:
1. Parse entity references from the memory sentence.
2. If no entities are referenced, insert as a standalone semantic node.
3. Otherwise, find existing semantic nodes connected to the same entities.
4. For each existing related node where the new memory's entities are a subset of the existing node's entities:
   - If cosine similarity > 0.85 (positive threshold): **reinforce** the existing node's edges (+1 weight) instead of inserting a duplicate. This is the "semantic memory consolidation" mechanism.
   - If cosine similarity < 0 (negative threshold): **weaken** the existing node's edges (-1 weight), representing contradictory information.
5. If no matching node was reinforced/weakened, create a new semantic node.

This process ensures that repeated observations strengthen confident knowledge, contradictory observations weaken uncertain knowledge, and genuinely new information is added.

---

## Control Pipeline

Entry point: `m3_agent/control.py`

The control pipeline answers questions about a video by iteratively reasoning over its VideoGraph memory.

### Multi-Turn Reasoning Loop

The loop runs for up to `total_round` (default 5) iterations per question, processing questions in batches of `batch_size` (default 64) for efficient GPU utilization.

**Initialization:**
- System prompt tells the model it has access to a memory bank and can either answer or search.
- First user message is an empty "Searched knowledge: {}".

**Each round:**
1. **LLM Generation:** The M3-Agent-Control model (loaded via vLLM with `tensor_parallel_size=2`) generates a response. The model uses Qwen's chat template with `enable_thinking=True`, producing chain-of-thought reasoning inside `<think>...</think>` tags followed by a structured action.

2. **Action Parsing:** The response is parsed for:
   - `Action: [Answer]` + `Content: <answer>` — terminates the loop.
   - `Action: [Search]` + `Content: <query>` — triggers memory retrieval.

3. **Memory Retrieval** (if search): The search query is passed to the retrieval module. Retrieved memories are formatted as "Searched knowledge: {CLIP_X: [memories...]}" and appended as the next user message.

4. **Forced Answer:** On the last round, the instruction explicitly tells the model it must output `[Answer]`.

**Special query handling:**
- Queries containing "character id" trigger `mem_wise=True` retrieval with `topk=20`, returning individual memory nodes rather than full clips.
- All other queries use clip-based retrieval with `topk` from config and a similarity threshold of 0.5.

### Retrieval Module

**File:** `mmagent/retrieve.py`

The retrieval module bridges natural language queries and the VideoGraph memory.

#### `retrieve_from_videograph(video_graph, query, topk, mode, threshold, before_clip)`

Core retrieval function:
1. **Back-translate** the query: replace character names (e.g., "character_0") with all possible face/voice IDs they map to, creating multiple query variants (capped at 100).
2. **Find related entity nodes** from the query's entity mentions.
3. **Embed** all query variants using `text-embedding-3-large`.
4. **Search** text nodes via `search_text_nodes()` — if entity nodes were found, search is restricted to text nodes connected to those entities; otherwise searches all text nodes.
5. **Aggregate** node scores into clip scores. Each clip's score is the max (or sum/mean) of its constituent node scores.
6. **Rank** clips by score, filter by threshold and `before_clip` constraint, return top-k.

#### `search(video_graph, query, current_clips, topk, ...)`

Higher-level search function that:
- Calls `retrieve_from_videograph` to get top clips.
- Filters out clips already seen (`current_clips`).
- For each new clip, collects all text node contents, applying `translate()` to replace internal IDs (face_X, voice_X) with unified character IDs (character_X).
- Returns `{CLIP_X: [memory_strings]}` dict.

In `mem_wise=True` mode, returns individual top-scoring nodes rather than full clips — used for targeted queries like character identity lookups.

#### Character ID Translation

- **`translate(video_graph, memories)`**: Replaces raw face/voice IDs with unified `character_X` IDs using `reverse_character_mappings`. Filters out raw "Equivalence:" statements (these are internal metadata, not useful in the answer).
- **`back_translate(video_graph, queries)`**: Inverse of translate. Replaces `character_X` IDs with all possible face/voice IDs, creating a combinatorial expansion of query variants. This ensures the embedding search can match memories that reference specific face/voice IDs.

#### Prompt-Based Retrieval (GPT-4o/Gemini variant)

**File:** `mmagent/retrieve.py` — `answer_with_retrieval()`

An alternative control loop using external API models (GPT-4o) instead of the fine-tuned M3-Agent-Control:
1. Optionally generates a retrieval plan using Gemini-1.5-Pro (`prompt_generate_plan`), given a video clip as context.
2. At each step, calls `generate_action()` which prompts GPT-4o with the question, accumulated knowledge, and retrieval plan.
3. The model outputs `[ANSWER]` or `[SEARCH]` with content.
4. Supports **route switching**: if a search returns no results, the next prompt uses a "new direction" variant that instructs the model to change its search strategy.
5. Supports **multiple queries**: generates 5 diverse queries per step, selects the one with lowest average cosine similarity to previous queries (maximizes diversity).

### Answer Evaluation

`eval_answer(question, predict, ground_truth)` in `control.py` sends the question, predicted answer, and ground truth to GPT-4o using `prompt_agent_verify_answer_referencing`. The evaluator checks whether the ground truth can be logically inferred from the predicted answer (not just exact match), returning "Yes" or "No".

---

## Prompt System

**File:** `mmagent/prompts.py` (~1000 lines)

All prompts are organized by function:

### Memorization Prompts

| Prompt | Used By | Purpose |
|--------|---------|---------|
| `prompt_audio_segmentation` | `voice_processing.py` (sent to Gemini) | ASR + speaker turn segmentation from video |
| `prompt_generate_captions_with_ids` | `memory_processing.py` (GPT-4o/Gemini) | Generate atomic episodic descriptions using face/voice IDs |
| `prompt_generate_thinkings_with_ids` | `memory_processing.py` (GPT-4o/Gemini) | Generate semantic conclusions (equivalences, character traits, relationships, plot, knowledge) |
| `prompt_generate_full_memory` | `memory_processing.py` (Gemini) | Combined episodic + semantic generation in one call |
| `prompt_generate_memory_with_ids_sft` | `memory_processing_qwen.py` (Qwen) | SFT-format combined episodic + semantic generation |
| `prompt_generate_captions_with_ids_sft` | (Qwen SFT training) | Simplified episodic caption prompt for fine-tuning |
| `prompt_generate_thinkings_with_ids_sft` | (Qwen SFT training) | Simplified semantic thinking prompt for fine-tuning |
| `prompt_generate_semantic_memory_with_ids_sft_*` | (Qwen SFT training) | Decomposed semantic prompts by category: equivalence, character, relation, plot, general_knowledge |

### Control Prompts

| Prompt | Used By | Purpose |
|--------|---------|---------|
| `prompt_generate_plan` | `retrieve.py` (Gemini) | Generate retrieval plan from video clip + question |
| `prompt_generate_action_with_plan` | `retrieve.py` (GPT-4o) | Reason + answer/search given question, knowledge, and plan |
| `prompt_generate_action_with_plan_new_direction` | `retrieve.py` (GPT-4o) | Same, but forces strategy change after empty retrieval |
| `prompt_generate_action_with_plan_multiple_queries` | `retrieve.py` (GPT-4o) | Generates 5 diverse queries instead of 1 |
| `prompt_generate_action_with_plan_multiple_queries_new_direction` | `retrieve.py` (GPT-4o) | Multiple queries + forced strategy change |
| `prompt_answer_with_retrieval_final` | `retrieve.py` / `control.py` | Force a final answer from accumulated knowledge |

### Evaluation Prompts

| Prompt | Used By | Purpose |
|--------|---------|---------|
| `prompt_agent_verify_answer_referencing` | `control.py`, `retrieve.py` | Check if ground truth is inferable from predicted answer |
| `prompt_agent_verify_answer` | (evaluation scripts) | Check semantic consistency between predicted and ground truth |
| `prompt_benchmark_verify_answer` | (benchmark evaluation) | Simple yes/no semantic match |

### Other Prompts

| Prompt | Purpose |
|--------|---------|
| `prompt_extract_entities` | Extract equivalence relationships from semantic memory |
| `prompt_refine_qa_list` | Translate/refine QA pairs from Chinese to English |
| `prompt_baseline_answer_clipwise_extract/summarize` | Baseline: extract info per clip then summarize |

---

## API & Inference Backends

### Azure OpenAI (`mmagent/utils/chat_api.py`)

Wraps Azure OpenAI for:
- **Chat completions** (GPT-4o, Gemini): `get_response()`, with retry logic and parallel batch processing via `ThreadPoolExecutor`.
- **Embeddings** (`text-embedding-3-large`): `get_embedding()`, with parallel batch processing.
- **Whisper** (audio transcription): `get_whisper()`.

All API calls respect per-model QPM (queries per minute) limits from `api_config.json`. Retries up to 5 times with 20-second backoff.

`generate_messages(inputs)` converts the internal multimodal format (text, images, video, audio as base64) into the OpenAI chat completion message format.

### Qwen2.5-Omni (`mmagent/utils/chat_qwen.py`)

Loads the fine-tuned M3-Agent-Memorization model locally:
- Uses `Qwen2_5OmniThinkerForConditionalGeneration` with Flash Attention 2.
- Lazy-loaded on first call (global `thinker` and `processor` singletons).
- Processes multimodal inputs (video path, images, text) via `qwen_omni_utils.process_mm_info`.
- Generates with `do_sample=True`, `temperature` from config (default 1e-6, nearly deterministic).
- CUDA memory is explicitly freed after each generation.

`generate_messages(inputs)` converts to Qwen's message format (notably: videos are passed by file path, not base64; images use `data:image;base64,` format).

### vLLM (`m3_agent/control.py`)

The M3-Agent-Control model is served via vLLM for efficient batched inference:
- `tensor_parallel_size=2` (requires 2 GPUs).
- `SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=1024)`.
- Uses Qwen's tokenizer with `enable_thinking=True` for chain-of-thought generation.

---

## Character Identity Resolution

One of M3-Agent's key innovations is resolving which face and voice nodes refer to the same character. This happens at multiple levels:

### 1. Perceptual Level (during memorization)
- Face embeddings are clustered within each clip (HDBSCAN).
- Across clips, faces and voices are matched to existing graph nodes via cosine similarity thresholds.

### 2. Semantic Level (LLM-generated equivalences)
- The memorization LLM generates "Equivalence: <face_X>, <voice_Y>" statements based on visual-auditory alignment (e.g., lips moving when voice is heard, timing of speech).
- These are stored as semantic nodes in the graph.

### 3. Graph Level (`refresh_equivalences()`)
- Union-Find merges all face/voice nodes that are transitively equivalent.
- Each equivalence group gets a unified `character_X` label.
- Conflicting equivalences (e.g., one voice matched to two faces) are resolved by `fix_collisions()`, keeping only the highest-weight association.

### 4. Query Level (translate / back-translate)
- At retrieval time, `character_X` in queries is expanded back to all constituent face/voice IDs.
- In retrieved memories, face/voice IDs are collapsed back to `character_X` for readability.

This multi-level approach allows the agent to reason about characters consistently even when the same person appears with different perceptual signatures across video clips.
