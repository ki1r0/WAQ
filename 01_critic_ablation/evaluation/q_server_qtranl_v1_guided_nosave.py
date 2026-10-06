#!/usr/bin/env python3
"""Eval bridge for legacy twin_language_query_transformer_v1 checkpoints.

The reward-ablation critics were trained with the online-Q language bundle,
while the current stock server registry only exposes the target-Q variant.
This bridge registers an adapter that uses the exact saved online twin Q heads;
it does not alter checkpoint weights or the evaluator protocol.
"""
from __future__ import annotations
import sys
from pathlib import Path
PI06_SOURCE = Path("/path/to/lingbot-va-original-rl-pi0.6")
if not PI06_SOURCE.is_dir():
    raise SystemExit(f"missing required pi0.6 source: {PI06_SOURCE}")
sys.path.insert(0, str(PI06_SOURCE))
from wan_va import wan_va_server as base_server  # noqa: E402
from wan_va import wan_va_server_q_guiding as q_server  # noqa: E402
from wan_va.rl import guidance as guidance_mod  # noqa: E402
from wan_va.rl.guidance import QGuidanceAdapter  # noqa: E402

class LegacyLanguageQGuidanceAdapter(QGuidanceAdapter):
    feature_mode = "language_query_context"
    def __init__(self, bundle):
        super().__init__()
        self.bundle = bundle
    def q_values(self, features, action_mask):
        del action_mask
        return self.bundle.q(
            features["q_context_tokens"],
            features["q_context_mask"],
            features["text_tokens"],
            features["text_mask"],
        )
    def objective(self, features, action_mask, mode="min"):
        q1, q2 = self.q_values(features, action_mask)
        return guidance_mod._select_q_objective(q1, q2, mode)

# The checkpoint's architecture and state dict remain the legacy v1 bundle;
# only the server-side adapter lookup is extended for this process.
guidance_mod.Q_GUIDANCE_REGISTRY["twin_language_query_transformer_v1"] = LegacyLanguageQGuidanceAdapter

def _skip_debug_artifact_save(_obj, _file_path):
    return None
base_server.save_async = _skip_debug_artifact_save
q_server.save_async = _skip_debug_artifact_save

if __name__ == "__main__":
    q_server.init_logger()
    q_server.main()
