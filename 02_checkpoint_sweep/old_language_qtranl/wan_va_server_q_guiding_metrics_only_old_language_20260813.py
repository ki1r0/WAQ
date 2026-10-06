#!/usr/bin/env python3
"""Metrics-only old Language QTran-L evaluation entrypoint.

The policy and Q-guidance computation are unchanged. Only asynchronous
serialization of observations, latents, and actions is disabled.
"""
import sys

OLD_LANGUAGE_SOURCE = "/path/to/mnt2/robotwin_language_qtranl_eval_20260812"
if OLD_LANGUAGE_SOURCE in sys.path:
    sys.path.remove(OLD_LANGUAGE_SOURCE)
sys.path.insert(0, OLD_LANGUAGE_SOURCE)

from wan_va import wan_va_server as base_server
from wan_va import wan_va_server_q_guiding as q_server


def _skip_debug_artifact_save(_obj, _file_path):
    return None


base_server.save_async = _skip_debug_artifact_save
q_server.save_async = _skip_debug_artifact_save


if __name__ == "__main__":
    q_server.init_logger()
    q_server.main()
