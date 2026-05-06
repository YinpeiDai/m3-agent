"""Stage B: assemble a per-video-chain VideoGraph from precomputed unit JSONs.

A "chain" is one ``video_chain_id`` (vc). Every SimLife task JSON sharing
the same vc has identical ``video_ids``; only ``stop_day_position`` and the
attached questions differ. Building one graph per chain (rather than per
task) saves ~62% Stage B work and storage. Per-task differences live on
the QA side as ``before_clip`` inside ``data/annotations/simlife.json``.

For each video unit referenced in chain["video_ids"] (in order), iterate
its clips; replay cached faces/voices/memories through the existing
``add_*_node`` / ``update_node`` machinery so node IDs are minted globally
within this chain's graph; then pickle the result.

We do NOT re-run face detection, speaker embedding, or Qwen captioning here —
all expensive work happened in stage A.
"""
import argparse
import collections
import glob
import json
import logging
import os
import pickle
import re
import sys
import time

from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mmagent.face_processing import (
    add_face_clusters_to_graph,
    establish_mapping,
    filter_score_based,
)
from mmagent.simlife_avatars import (
    avatar_face_info,
    CHARACTER_NAMES,
    FIXED_ID_TO_NAME,
    N_FIXED,
)
from mmagent.simlife_voice_processing import update_videograph_from_cache
from mmagent.memory_processing_qwen import process_memories
from mmagent.videograph import VideoGraph

logger = logging.getLogger(__name__)

CLIPS_PER_UNIT_OFFSET = 100  # must match m3_agent/simlife_data_prep.py

# Matches the same id-bearing tokens parse_video_caption looks for, but only
# for the entity types we know how to remap. <character_*> tokens, if any,
# are left untouched (refresh_equivalences handles those after assembly).
_FACE_RE = re.compile(r"<face_(\d+)>")
_VOICE_RE = re.compile(r"<voice_(\d+)>")


def _count_clips(unit_inter_dir):
    return len(glob.glob(os.path.join(unit_inter_dir, "clip_*_faces.json")))


def _process_clip_faces(graph, face_path):
    """Stage-B per-clip face replay. Returns the lossless cluster_id ->
    graph_node_id map (process_faces alone can't surface this — its return
    value applies a top-K cap that may discard cluster_ids when several
    clusters merge into one node).

    Always extends the result with identity mappings for the four avatar
    slots ``0..3``: those graph nodes were seeded at chain start with
    fixed ids, so any ``<face_K>`` reference (K in 0..3) — including the
    derived ``Equivalence`` lines from precompute — resolves even when
    that character's face doesn't visually appear in this clip.
    """
    cluster_to_node = {}
    if os.path.exists(face_path):
        with open(face_path) as f:
            faces_json = json.load(f)
        if faces_json:
            tempid2faces = establish_mapping(
                faces_json, key="cluster_id", filter=filter_score_based,
            )
            if tempid2faces:
                _, cluster_to_node = add_face_clusters_to_graph(graph, tempid2faces)

    # Pin the four avatar slots: cluster_id K (K in 0..3) always points
    # at graph node K (the seeded avatar). setdefault preserves whichever
    # graph node id add_face_clusters_to_graph already chose for an
    # actually-present cluster (typically the seeded id, since avatar
    # cosine match is high).
    for k in range(N_FIXED):
        cluster_to_node.setdefault(k, k)
    return cluster_to_node


def _build_voice_local_to_global(voice_entries, speaker_order=None):
    """Build ``{local_voice_id: graph_node_id}`` for one clip.

    Per-utterance convention: the i-th entry in ``voice_entries`` has
    clip-local voice id ``i``; we map it to whichever graph voice node
    ``update_videograph_from_cache`` matched/created for that utterance.
    Two entries by the same speaker often share a ``matched_node``
    (embedding cosine ≥ audio_matching_threshold), so distinct local ids
    can map to the same graph node — that's how same-speaker references
    coalesce in the rewritten memory text.

    ``speaker_order`` is retained for backward-compat with legacy memory
    JSONs that saved an explicit speaker positional order. When
    provided, the i-th *speaker* in that order claims local id ``i`` and
    we look up the first matching utterance's ``matched_node``. New
    JSONs don't carry the field; in that case we use per-utterance ids.

    ``voice_entries`` should be the in-memory list AFTER
    ``update_videograph_from_cache`` has populated ``matched_node`` on
    each entry.
    """
    out = {}
    if speaker_order:
        # Legacy positional convention: local id == speaker first-appearance index.
        by_speaker = collections.defaultdict(list)
        for entry in voice_entries:
            by_speaker[entry.get("speaker") or "unknown"].append(entry)
        for idx, spk in enumerate(speaker_order):
            entries = by_speaker.get(spk, [])
            if entries and "matched_node" in entries[0]:
                out[idx] = int(entries[0]["matched_node"])
        return out

    # Per-utterance: i-th entry -> i.
    for i, entry in enumerate(voice_entries):
        if "matched_node" in entry:
            out[i] = int(entry["matched_node"])
    return out


def _rewrite_ids(text, face_map, voice_map):
    def face_sub(m):
        local = int(m.group(1))
        if local in face_map:
            return f"<face_{face_map[local]}>"
        return ""  # drop unresolved placeholders so parse_video_caption ignores them

    def voice_sub(m):
        local = int(m.group(1))
        if local in voice_map:
            return f"<voice_{voice_map[local]}>"
        return ""

    text = _FACE_RE.sub(face_sub, text)
    text = _VOICE_RE.sub(voice_sub, text)
    return text


def _filter_and_rewrite(raw_texts, raw_embeddings, face_map, voice_map):
    """Drop empty raw entries (matching the original ``if t`` filter),
    rewrite local ids in the kept ones, and **keep the embedding list
    aligned**.

    Returns ``(texts, embeddings)``. ``embeddings`` is ``None`` when the
    caller didn't pass cached embeddings or when alignment broke (length
    mismatch on input); ``process_memories`` then falls back to the API.

    The cached embedding belongs to the *raw* text; rewriting only swaps
    integers in ``<face_X>``/``<voice_Y>`` placeholders, so the embedding
    of "<face_0> walks in" is functionally equivalent to that of
    "<face_5> walks in" for retrieval. We don't re-embed.
    """
    texts = []
    embs = [] if raw_embeddings is not None else None
    if raw_embeddings is not None and len(raw_embeddings) != len(raw_texts):
        # Alignment is broken at the source — silently fall back.
        embs = None
    for i, t in enumerate(raw_texts):
        if not t:
            continue
        texts.append(_rewrite_ids(t, face_map, voice_map))
        if embs is not None:
            embs.append(raw_embeddings[i])
    return texts, embs


def _memory_filename(k, variant):
    """Two memory variants are written by Stage A4:
        clip_K_memory_audio.json    — Qwen with audio modality + voice JSON
        clip_K_memory_noaudio.json  — Qwen with audio off + voices stripped
    Legacy single-variant ``clip_K_memory.json`` is treated as the audio
    variant for backward compat (older units precomputed before the split).
    """
    return f"clip_{k}_memory_{variant}.json"


def _resolve_clip_paths(unit_dir, override_dir, k, variant):
    """Return (face_path, voice_path, mem_path) for clip K under ``variant``
    (``'audio'`` or ``'noaudio'``).

    Faces always come from the per-unit default (overrides only change
    audio/text). Voices: under ``variant='audio'`` we use the per-chain
    override pair when both override files exist (atomic pairing), else
    fall back to defaults. Under ``variant='noaudio'`` the override flow
    doesn't apply at all — vision-only memory ignores audio, so the
    per-unit default voice + default noaudio memory is always used.
    """
    face_path = os.path.join(unit_dir, f"clip_{k}_faces.json")
    default_voice = os.path.join(unit_dir, f"clip_{k}_voices.json")
    default_mem = os.path.join(unit_dir, _memory_filename(k, variant))
    legacy_mem = os.path.join(unit_dir, f"clip_{k}_memory.json")
    # Legacy fallback: if the variant-suffixed file is missing but the
    # single-file legacy memory exists, use it (only meaningful for
    # variant='audio').
    if (not os.path.exists(default_mem) and variant == "audio"
            and os.path.exists(legacy_mem)):
        default_mem = legacy_mem

    if variant == "audio" and override_dir:
        ov_voice = os.path.join(override_dir, f"clip_{k}_voices.json")
        ov_mem_var = os.path.join(override_dir, _memory_filename(k, "audio"))
        ov_mem_legacy = os.path.join(override_dir, f"clip_{k}_memory.json")
        ov_mem = ov_mem_var if os.path.exists(ov_mem_var) else ov_mem_legacy
        if os.path.exists(ov_voice) and os.path.exists(ov_mem):
            return face_path, ov_voice, ov_mem
    return face_path, default_voice, default_mem


def _seed_avatar_face_nodes(graph):
    """Insert one face node per available avatar BEFORE any clip is
    processed. Because ``VideoGraph.next_node_id`` starts at 0, this fixes:

        graph node 0 = Father Sim
        graph node 1 = Mother Sim
        graph node 2 = Son Sim
        graph node 3 = Daughter Sim

    Per-clip clusters whose unit-level ``cluster_id`` was already pinned to
    these integers by ``recluster_unit_faces`` will then collide with the
    seeds via ``search_img_nodes`` (cosine similarity of the avatar
    embedding against the cluster's embedding). On match, the graph
    *extends* the seed node; otherwise it creates a new one — typical only
    when avatar matching at the unit-level recluster failed too, e.g. for
    strongly off-pose detections.

    Avatars whose face InsightFace couldn't detect on avatars.png are
    skipped silently; the corresponding graph node id is then minted by
    the first matching cluster instead, with no fixed-id guarantee for
    that character.
    """
    pairs = avatar_face_info()  # [(fixed_id, name, face_info), ...]
    if not pairs:
        return []
    seeded = []
    for fixed_id, name, face_info in pairs:
        nid = graph.add_img_node(face_info)
        if nid != fixed_id:
            # Should never happen on a fresh graph; log if it does because
            # downstream assumptions break otherwise.
            logger.warning(
                "Avatar %s expected node id %d but got %d; downstream fixed-id "
                "assumptions may break", name, fixed_id, nid,
            )
        seeded.append((nid, name))
    logger.info("Seeded %d avatar face nodes: %s",
                len(seeded), ", ".join(f"{nid}={n}" for nid, n in seeded))
    return seeded


def assemble(chain_row, memory_config, inter_root="data/intermediate", variant="audio"):
    """Build a VideoGraph for one chain in either ``audio`` or ``noaudio``
    variant. The two variants share faces and voice JSONs but differ in
    which memory text gets ingested:

      audio   — ``clip_K_memory_audio.json``   (Qwen with audio modality on)
      noaudio — ``clip_K_memory_noaudio.json`` (Qwen with audio modality off,
                                                voice JSON stripped from prompt)

    The override flow only touches the audio variant — noaudio memory
    is identical across chains using the same unit, so it always reads
    from the per-unit default.
    """
    graph = VideoGraph(**memory_config)
    chain_id = chain_row.get("chain_id") or "?"

    # Pre-scan the chain so the progress bar has a real total and we can
    # warn about missing units up-front instead of mid-iteration.
    units_to_process = []
    total_clips = 0
    for unit_id in chain_row["video_ids"]:
        unit_dir = os.path.join(inter_root, unit_id)
        if not os.path.isdir(unit_dir):
            logger.warning("[%s] missing intermediate dir for %s; skipping", chain_id, unit_id)
            continue
        n = _count_clips(unit_dir)
        if n == 0:
            logger.warning("[%s] no face JSONs in %s; skipping", chain_id, unit_dir)
            continue
        units_to_process.append((unit_id, unit_dir, n))
        total_clips += n
    logger.info("[%s] %s variant: %d units, %d clips total",
                chain_id, variant, len(units_to_process), total_clips)

    # Seed the four SimLife mains at fixed graph node ids before any clip
    # is processed. This way <face_0..3> mean Father/Mother/Son/Daughter in
    # every chain pickle, matching the fixed cluster_ids that
    # recluster_unit_faces assigns.
    _seed_avatar_face_nodes(graph)
    # Voice nodes are NOT pre-seeded: voice ids in the prompt are
    # per-utterance (one ``<voice_X>`` per piece of speech). Same-speaker
    # utterances still merge into one graph voice node via cosine
    # similarity inside ``update_videograph_from_cache``, so distinct
    # prompt-side ``<voice_X>``s typically rewrite to the same graph
    # node id — but the graph node ids themselves are dynamic per chain.

    pbar = tqdm(total=total_clips,
                desc=f"{chain_id} ({variant})",
                unit="clip",
                dynamic_ncols=True)
    chain_t0 = time.time()
    for unit_idx, (unit_id, unit_dir, n_clips) in enumerate(units_to_process):
        override_dir = None
        if chain_id and chain_id != "?":
            candidate = os.path.join(inter_root, "per_chain", chain_id, unit_id)
            if os.path.isdir(candidate):
                override_dir = candidate

        unit_t0 = time.time()
        unit_start_nodes = len(graph.nodes)
        unit_start_text = sum(len(v) for v in graph.text_nodes_by_clip.values())
        used_override_clips = 0
        for k in range(n_clips):
            face_path, voice_path, mem_path = _resolve_clip_paths(
                unit_dir, override_dir, k, variant,
            )
            if override_dir and os.path.dirname(mem_path) == override_dir:
                used_override_clips += 1

            # Faces — replay cached cluster_ids through the graph and capture
            # the local->global label map before any top-K cap clobbers it.
            face_map = _process_clip_faces(graph, face_path)

            # Voices — load JSON ourselves so we can read matched_node back per entry.
            voice_map = {}
            if os.path.exists(voice_path):
                with open(voice_path) as f:
                    voice_entries = json.load(f)
                if voice_entries:
                    update_videograph_from_cache(graph, voice_entries)
                    # voice_speaker_order survives in *legacy* memory JSONs
                    # written before voice anchoring; if present, we honour
                    # the positional convention from that era. New JSONs
                    # don't carry the field — we derive the local-id map
                    # from speaker names directly.
                    legacy_order = []
                    if os.path.exists(mem_path):
                        try:
                            legacy_order = json.load(open(mem_path)).get(
                                "voice_speaker_order", []) or []
                        except Exception:
                            legacy_order = []
                    voice_map = _build_voice_local_to_global(
                        voice_entries,
                        speaker_order=legacy_order if legacy_order else None,
                    )

            # Memory — rewrite ids, then process via existing pipeline.
            # Cached embeddings (saved at Stage A4) are passed through so
            # process_memories doesn't have to re-fetch from the
            # text-embedding-3-large API on every chain rebuild.
            if not os.path.exists(mem_path):
                pbar.update(1)
                continue
            mem = json.load(open(mem_path))
            episodic_raw = mem.get("episodic", []) or []
            semantic_raw = mem.get("semantic", []) or []
            episodic, epi_embs = _filter_and_rewrite(
                episodic_raw, mem.get("episodic_embeddings"), face_map, voice_map,
            )
            semantic, sem_embs = _filter_and_rewrite(
                semantic_raw, mem.get("semantic_embeddings"), face_map, voice_map,
            )

            clip_id = unit_idx * CLIPS_PER_UNIT_OFFSET + k
            if episodic:
                process_memories(graph, episodic, clip_id,
                                 type="episodic", embeddings=epi_embs)
            if semantic:
                process_memories(graph, semantic, clip_id,
                                 type="semantic", embeddings=sem_embs)

            pbar.update(1)
            pbar.set_postfix(
                unit=f"{unit_idx + 1}/{len(units_to_process)}",
                clip=f"{k + 1}/{n_clips}",
                nodes=len(graph.nodes),
                text=sum(len(v) for v in graph.text_nodes_by_clip.values()),
                edges=len(graph.edges),
                refresh=False,
            )

        unit_elapsed = time.time() - unit_t0
        added_nodes = len(graph.nodes) - unit_start_nodes
        added_text = sum(len(v) for v in graph.text_nodes_by_clip.values()) - unit_start_text
        logger.info(
            "[%s] unit %d/%d %s done in %.1fs (+%d nodes, +%d text nodes)%s",
            chain_id, unit_idx + 1, len(units_to_process), unit_id,
            unit_elapsed, added_nodes, added_text,
            f" [override clips: {used_override_clips}]" if used_override_clips else "",
        )
    pbar.close()

    refresh_t0 = time.time()
    logger.info("[%s] running refresh_equivalences()...", chain_id)
    graph.refresh_equivalences()
    logger.info(
        "[%s] %s assembled in %.1fs (refresh_equivalences=%.1fs): "
        "nodes=%d edges=%d characters=%d",
        chain_id, variant, time.time() - chain_t0, time.time() - refresh_t0,
        len(graph.nodes), len(graph.edges),
        len(getattr(graph, "character_mappings", {})),
    )
    return graph


def _variant_pkl_path(base_mem_path, variant):
    """For ``data/memory_graphs/vc_NNN.pkl`` and variant ``noaudio`` return
    ``data/memory_graphs/vc_NNN_noaudio.pkl``; ``audio`` likewise. Always
    suffixes so the two variants never share a path."""
    if base_mem_path.endswith(".pkl"):
        return f"{base_mem_path[:-4]}_{variant}.pkl"
    return f"{base_mem_path}_{variant}.pkl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", help="chain_id basename, e.g. vc_000001")
    parser.add_argument("--data_chains", default="data/simlife/data_chains.jsonl")
    parser.add_argument("--inter_root", default="data/intermediate")
    parser.add_argument("--memory_config", default="configs/memory_config.json")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--variant", choices=["audio", "noaudio", "both"],
                        default="both",
                        help="Which memory variant to build. Default builds "
                             "both audio (vc_NNN_audio.pkl) and noaudio "
                             "(vc_NNN_noaudio.pkl) graphs.")
    args = parser.parse_args()

    # mmagent/__init__.py installs root handlers (or sets level to CRITICAL
    # in training mode), so basicConfig() is a no-op here. Replace root
    # handlers with one stderr handler at the requested level so progress
    # logging is always visible without doubled output. tqdm will continue
    # to write its bar to stderr alongside.
    level = getattr(logging, args.log_level.upper())
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.setLevel(level)
    root.addHandler(handler)

    memory_config = json.load(open(args.memory_config))

    chain_row = None
    with open(args.data_chains) as f:
        for line in f:
            row = json.loads(line)
            if args.chain is None or row["chain_id"] == args.chain:
                chain_row = row
                break
    if chain_row is None:
        raise SystemExit(f"Chain {args.chain!r} not found in {args.data_chains}")

    base_mem_path = chain_row["mem_path"]
    os.makedirs(os.path.dirname(base_mem_path), exist_ok=True)

    variants = ["audio", "noaudio"] if args.variant == "both" else [args.variant]
    for variant in variants:
        out_path = _variant_pkl_path(base_mem_path, variant)
        if os.path.exists(out_path) and not args.overwrite:
            logger.info("Already exists: %s (use --overwrite to rebuild)", out_path)
            continue
        graph = assemble(chain_row, memory_config, inter_root=args.inter_root,
                         variant=variant)
        with open(out_path, "wb") as f:
            pickle.dump(graph, f)
        logger.info("wrote %s (variant=%s nodes=%d edges=%d characters=%d)",
                    out_path, variant, len(graph.nodes), len(graph.edges),
                    len(getattr(graph, "character_mappings", {})))


if __name__ == "__main__":
    main()
