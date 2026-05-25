# Official Artifacts

These files are part of the fixed main-performance reproduction path.

- `p2c_outputs/*.json`: original P2C Generator outputs used to build answer-field
  embeddings without rerunning the LLM.
- `frame_embeddings/*.npz`: original frame feature vectors, stored as compressed
  release artifacts and materialized to `outputs/embeddings/` by
  `scripts/embed_inputs.py --only frames`.

Reader-generated files should go under `outputs/`, not here.
