"""Reuse the verified task-4 10k evaluation, changing only checkpoint/host/output/ports."""
from pathlib import Path

source = Path('/mnt/zengqixin/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/A_line2/launch_va2_noq_mugs_2078_step10000.py')
script = source.read_text()
script = script.replace('step10000', 'step15000').replace('STEP10000', 'STEP15000')
# Preserve the actual export directory name; it contains eval2078 for historical reasons.
script = script.replace('lyg2078-va2', 'lyg2026-va2')
script = script.replace('MUGS-2078-', 'MUGS-2026-')
script = script.replace('2078_step15000', '2026_step15000')
assert 'PORT_BASE="34720"' in script
script = script.replace('PORT_BASE="34720"', 'PORT_BASE="35720"')
exec(compile(script, str(source), 'exec'))
