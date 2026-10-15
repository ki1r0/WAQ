#!/usr/bin/env python3
"""VA2 step 7500, LIBERO-10, no critic; repeat each action twice at 20Hz.

python eval_libero_va2_noq_10hz_20260911.py --gpus 0,1,2,3,4,5,6,7
Add --tasks 4 --trials 10 for a small test, --save-video for 10Hz videos.
Use --action-repeat 1 in a separate output directory for the 20Hz control.
--dry-run checks paths and shows assignments; --self-test needs no GPU.
Only new output directories are accepted. Original code/data remain untouched.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime


ROOT = Path('/home/zengqixin/wy/lingbot-va-original-rl-adaptive-cpb')
OVERLAY = '/home/zengqixin/wy/lingbot-eval-deps/lib/python3.10/site-packages'
LIBERO = '/home/zengqixin/wy/LIBERO'
POLICY_PYTHON = '/home/zengqixin/miniconda3/envs/lingbot/bin/python'
CLIENT_PYTHON = '/home/zengqixin/miniconda3/envs/liberoplus/bin/python'
CHECKPOINT = '/mnt/zengqixin/experiments/robotwin_clean_sft_20260903/export/from_va2_gripperfix_20260909_step7500_diffusers'


def rollout(model, env, first_obs, action_repeat, max_steps, extract_obs, video=None):
    """Keep temporal actions (7, 4, 4); sample RGB after each repeated command."""
    import numpy as np

    first = True
    while env.env.timestep < max_steps:
        # The pure VA server uses model-space zeros for initial action padding.
        action = np.asarray(model.infer({'obs': first_obs})['action'])
        if action.shape != (7, 4, 4) or not np.isfinite(action).all():
            raise ValueError(f'Invalid temporal action from server: {action.shape}')
        frames = []
        for frame in range(1 if first else 0, 4):
            for slot in range(4):
                for _ in range(action_repeat):
                    if env.env.timestep >= max_steps:
                        return False
                    obs, _, done, _ = env.step(action[:, frame, slot])
                    if done:
                        if video is not None:
                            video.append(extract_obs(obs))
                        return True
                observed = extract_obs(obs)
                frames.append(observed)
                if video is not None:
                    video.append(observed)
        if env.env.timestep >= max_steps:
            return False
        # Four RGBs first, eight thereafter. Do not duplicate action KV tokens.
        model.infer(dict(obs=frames, state=action, compute_kv_cache=True, imagine=False))
        first = False
    return False


def server(config, worker):
    sys.path.insert(0, str(ROOT))
    import torch
    from wan_va import wan_va_server as va
    from wan_va.utils.Simple_Remote_Infer.deploy.websocket_policy_server import WebsocketPolicyServer

    class SeededVA(va.VA_Server):
        def infer(self, request):
            if request.get('reset'):
                torch.manual_seed(int(request['seed']))
            return super().infer(request)

    va.init_logger()
    torch.cuda.set_device(0)
    cfg = copy.deepcopy(va.VA_CONFIGS['libero'])
    cfg.frame_chunk_size = 4
    cfg.rank = cfg.local_rank = 0
    cfg.world_size = 1
    cfg.infer_mode = 'server'
    cfg.wan22_pretrained_model_name_or_path = config['checkpoint']
    cfg.wan22_finetuned_model_name_or_path = config['checkpoint']
    cfg.save_root = str(Path(config['output']) / 'server' / f'worker-{worker}')
    Path(cfg.save_root).mkdir(parents=True)
    (Path(cfg.save_root) / 'config.json').write_text(json.dumps(cfg, indent=2, default=str))
    # Local process only: avoid the base server's per-chunk tensor/observation dumps.
    va.save_async = lambda *args, **kwargs: None
    model = SeededVA(cfg, robotwin_eval=True)
    metadata = dict(
        inference_protocol='original_temporal_action_kv',
        frame_chunk_size=4, action_per_frame=4, q_guidance=None,
        critic_loaded=False, policy_checkpoint=config['checkpoint'],
        output_root=config['output'], initial_action_conditioning='model_space_zero',
    )
    WebsocketPolicyServer(model, host='127.0.0.1',
                          port=config['workers'][worker]['port'], metadata=metadata).serve_forever()


def client(config, worker):
    sys.path.insert(0, str(ROOT))
    import numpy as np

    # Reuse task identities, camera preprocessing and atomic episode record writes.
    spec = importlib.util.spec_from_file_location('libero_original_client', ROOT / 'evaluation/libero/client.py')
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    suite = legacy.benchmark.get_benchmark_dict()['libero_10']()
    if suite.get_num_tasks() != 10:
        raise RuntimeError('Expected official LIBERO-10; check PYTHONPATH, not LIBERO-Plus task ordering')
    assignment = config['workers'][worker]
    model = legacy.WebsocketClientPolicy(host='127.0.0.1', port=assignment['port'])
    metadata = model.get_server_metadata()
    expected = dict(inference_protocol='original_temporal_action_kv', frame_chunk_size=4,
                    action_per_frame=4, critic_loaded=False, q_guidance=None,
                    policy_checkpoint=config['checkpoint'], output_root=config['output'])
    if any(metadata.get(key, 'missing') != value for key, value in expected.items()):
        raise RuntimeError(f'Unexpected server metadata: {metadata}')
    metadata.update(action_repeat=config['action_repeat'], sim_control_hz=20,
                    observation_hz=20 / config['action_repeat'], max_env_steps=config['max_steps'])
    output = Path(config['output']) / 'client' / f'worker-{worker}'
    for task_id, trial in assignment['episodes']:
        task = suite.get_task(task_id)
        states = suite.get_task_init_states(task_id)
        if not 0 <= trial < len(states):
            raise ValueError(f'Trial {trial} outside task {task_id} initial states')
        seed = config['seed'] + task_id * 1000 + trial
        identity = legacy._task_identity('libero_10', suite, task_id, trial, states,
                                        Path(config['checkpoint']).name, config['checkpoint'],
                                        f'va2-noq-frame4-repeat{config["action_repeat"]}',
                                        Path(config['output']).name)
        identity.update(seed=seed, action_repeat=config['action_repeat'])
        record = output / 'task_records/libero_10' / f'task_{task_id:05d}' / f'trial_{trial:03d}.json'
        if record.exists():
            raise FileExistsError(record)
        np.random.seed(seed)
        env = legacy.construct_single_env(dict(bddl_file_name=identity['bddl_path'],
                                              camera_heights=128, camera_widths=128, control_freq=20))
        if env is None:
            raise RuntimeError(f'Failed to create task {task_id}')
        try:
            env.seed(seed)
            first_obs, _ = legacy.init_single_env(env, states[trial])
            model.infer(dict(reset=True, prompt=task.language, seed=seed))
            video = [] if config['save_video'] else None
            success = rollout(model, env, first_obs, config['action_repeat'],
                              config['max_steps'], legacy._extract_obs, video)
            steps = env.env.timestep
        finally:
            env.close()
        video_path = ''
        if video:
            path = output / 'videos' / f'task_{task_id:02d}_trial_{trial:03d}_{success}.mp4'
            path.parent.mkdir(parents=True, exist_ok=True)
            legacy.save_video(video, path, fps=20 / config['action_repeat'])
            video_path = str(path)
        legacy._write_evaluation_record(output, identity, success, steps, video_path, metadata)
        print(f'task={task_id} trial={trial} success={success} env_steps={steps}', flush=True)
    model._ws.close()


def launch(args):
    gpus = [int(gpu) for gpu in args.gpus.split(',')]
    if not gpus or min(gpus) < 0 or len(gpus) != len(set(gpus)):
        raise ValueError('--gpus must be distinct non-negative GPU indices')
    if len(set(args.tasks)) != len(args.tasks) or not all(0 <= task < 10 for task in args.tasks):
        raise ValueError('--tasks must be distinct LIBERO-10 indices (0..9)')
    if args.trials < 1 or args.trial_offset < 0 or args.trial_offset + args.trials > 50:
        raise ValueError('Trials must stay within the 50 initial states per task')
    if args.max_steps <= 5 or not 0 <= args.seed < 2**32 - 10050:
        raise ValueError('--max-steps must exceed the 5 settling steps; seed must fit uint32')
    episodes = [(task, trial) for task in args.tasks
                for trial in range(args.trial_offset, args.trial_offset + args.trials)]
    gpus = gpus[:len(episodes)]
    if not 1024 <= args.port_base <= 65536 - len(gpus):
        raise ValueError('Invalid port range')
    checkpoint = args.checkpoint.resolve()
    required = [checkpoint / name for name in ('transformer', 'vae', 'tokenizer', 'text_encoder')]
    required += [checkpoint / 'transformer/diffusion_pytorch_model.safetensors',
                 ROOT / 'wan_va/wan_va_server.py', ROOT / 'evaluation/libero/client.py',
                 Path(POLICY_PYTHON), Path(CLIENT_PYTHON)]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    policy_label = checkpoint.name.removeprefix('from_').removesuffix('_diffusers')
    output = (args.output or Path('/mnt/zengqixin') /
              f'eval_libero10_{policy_label}_noq_repeat{args.action_repeat}_{datetime.now():%Y%m%d_%H%M%S}').resolve()
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    config = dict(checkpoint=str(checkpoint), output=str(output), action_repeat=args.action_repeat,
                  max_steps=args.max_steps, seed=args.seed, save_video=args.save_video,
                  workers=[dict(gpu=gpu, port=args.port_base + i, episodes=episodes[i::len(gpus)])
                           for i, gpu in enumerate(gpus)])
    print(json.dumps({**config, 'workers': [dict(gpu=w['gpu'], port=w['port'], episodes=len(w['episodes']))
                                          for w in config['workers']]}, indent=2), flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=False)
    logs = output / 'logs'
    logs.mkdir()
    (output / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    sources = [Path(__file__).resolve(), ROOT / 'wan_va/wan_va_server.py', ROOT / 'evaluation/libero/client.py']
    (output / 'source_sha256.json').write_text(json.dumps(
        {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}, indent=2))
    processes = []

    def start(mode, index):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpus[index]), MUJOCO_EGL_DEVICE_ID=str(gpus[index]),
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   TOKENIZERS_PARALLELISM='false', MUJOCO_GL='egl', PYTHONUNBUFFERED='1',
                   PYTHONPATH=f'{ROOT}:{OVERLAY}:{LIBERO}')
        python = POLICY_PYTHON if mode == 'server' else CLIENT_PYTHON
        command = [python, str(Path(__file__).resolve()), '--mode', mode,
                   '--output', str(output), '--worker', str(index)]
        with (logs / f'{mode}-{index}.log').open('x') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
        return process

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        servers = [start('server', i) for i in range(len(gpus))]
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        pending = set(range(len(gpus)))
        deadline = time.monotonic() + 900
        while pending:
            for i in list(pending):
                if servers[i].poll() is not None:
                    raise RuntimeError(f'Server {i} exited; see {logs / f"server-{i}.log"}')
                try:
                    with opener.open(f'http://127.0.0.1:{config["workers"][i]["port"]}/healthz', timeout=1) as response:
                        if response.status == 200:
                            pending.remove(i)
                except OSError:
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError(f'Server startup timed out: {sorted(pending)}; see {logs}')
            if pending:
                time.sleep(1)
        clients = [start('client', i) for i in range(len(gpus))]
        while True:
            codes = [process.poll() for process in clients]
            for i, code in enumerate(codes):
                if code not in (None, 0) or (code is None and servers[i].poll() is not None):
                    raise RuntimeError(f'Worker {i} failed; see {logs}')
            if all(code == 0 for code in codes):
                break
            time.sleep(1)
        rows = [json.loads(path.read_text()) for path in (output / 'client').rglob('trial_*.json')]
        identities = {(row['task_index'], row['trial_index']) for row in rows}
        if len(rows) != len(episodes) or identities != set(episodes):
            raise RuntimeError('Completed episode identities do not match the requested evaluation')
        successes = sum(row['success'] for row in rows)
        summary = dict(successes=successes, episodes=len(rows), success_rate=successes / len(rows),
                       per_task={str(task): dict(successes=sum(r['success'] for r in rows if r['task_index'] == task),
                                                 episodes=args.trials) for task in args.tasks})
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        (output / 'COMPLETE').write_text('All requested episodes completed.\n')
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def self_test():
    import numpy as np
    from types import SimpleNamespace

    class Env:
        def __init__(self, success_at=None):
            self.env = SimpleNamespace(timestep=0)
            self.success_at = success_at
            self.actions = []

        def step(self, action):
            self.actions.append(float(action[0]))
            self.env.timestep += 1
            return self.env.timestep, 0, self.env.timestep == self.success_at, {}

    class Model:
        def __init__(self):
            self.cache = []

        def infer(self, request):
            if request.get('compute_kv_cache'):
                self.cache.append(request)
                return {}
            assert 'state' not in request
            return {'action': np.broadcast_to(np.arange(16).reshape(1, 4, 4), (7, 4, 4))}

    for repeat in (1, 2):
        env, model = Env(), Model()
        assert not rollout(model, env, 0, repeat, 12 * repeat + 1, lambda obs: obs)
        assert env.env.timestep == 12 * repeat + 1
        assert env.actions[:4 * repeat] == [value for value in range(4, 8) for _ in range(repeat)]
        assert [r['obs'] for r in model.cache] == [list(range(repeat, 4 * repeat + 1, repeat)),
                                                 list(range(5 * repeat, 12 * repeat + 1, repeat))]
        assert all(r['state'].shape == (7, 4, 4) for r in model.cache)
    env, model = Env(success_at=1), Model()
    assert rollout(model, env, 0, 2, 800, lambda obs: obs)
    assert len(env.actions) == 1 and not model.cache
    print('PASS: repeat 1/2, initial padding, RGB cadence, action KV shape, horizon, early success')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, default=Path(CHECKPOINT))
    parser.add_argument('--gpus', default='0', help='Physical GPU IDs, comma separated')
    parser.add_argument('--tasks', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--trial-offset', type=int, default=0)
    parser.add_argument('--action-repeat', type=int, choices=(1, 2), default=2)
    parser.add_argument('--max-steps', type=int, default=800, help='Physical environment steps, including 5 settling steps')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--port-base', type=int, default=33700)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--mode', choices=('launch', 'server', 'client', 'test'), default='launch', help=argparse.SUPPRESS)
    parser.add_argument('--worker', type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_test:
        raise SystemExit(subprocess.run([POLICY_PYTHON, str(Path(__file__).resolve()), '--mode', 'test']).returncode)
    if args.mode == 'test':
        self_test()
    elif args.mode == 'launch':
        launch(args)
    else:
        config = json.loads((args.output / 'run_config.json').read_text())
        (server if args.mode == 'server' else client)(config, args.worker)


if __name__ == '__main__':
    main()
