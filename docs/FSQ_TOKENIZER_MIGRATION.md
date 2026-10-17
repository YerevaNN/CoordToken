# Current 3DMolGen integration

The atom-plus-register text interface is implemented in [3DMolGen's combined handoff branch](https://github.com/YerevaNN/3DMolGen/tree/coordtoken-integration-handoff). See its [integration guide](https://github.com/YerevaNN/3DMolGen/blob/coordtoken-integration-handoff/docs/FSQ_TOKENIZER_MIGRATION.md) for encoding, Qwen preparation/training/export, and decoding.

The format is `atom12_reg1_prefix_v1`: one molecular code first, followed by the molecular string with one geometry code after each atom. All codes use the same 4,096-token vocabulary. There is no `<REG>` marker. Non-atom punctuation has no geometry code. This replaces the earlier migration proposal.

The selected model is the two-epoch mixed-corpus checkpoint; [UPDATED_TOKENIZER.md](UPDATED_TOKENIZER.md) gives its stable shareable path and SHA-256. Original training-source snapshots remain unchanged. GPU production integration and full-corpus Qwen encoding are not claimed complete by publishing the adapter.
