"""
cc_selector.py — DDQN ZeroMQ 서버
---------------------------------
  • REP 소켓으로 tcp://*:5555 수신
  • 메시지 타입
      1) {type:"infer", pfMetric:[...], prevMask:[...]}
         → maskIdx + Q‑values 반환
      2) {type:"train", exp:{state,action,reward,next_state,done}}
         → 리플레이 버퍼 저장 + 한 번의 DDQN 업데이트
  • Ctrl +C 로 언제든 종료 (poll + KeyboardInterrupt)
"""
import os, sys, json, signal
import zmq
import numpy as np
import torch
import torch.optim as optim
import atexit
import matplotlib.pyplot as plt

training_rewards = [] # reward 모음
training_actions = [] # maskIdx 모음

def _save_training_history():
    # 1) action/reward 배열 합치기
    data = np.column_stack([training_actions, training_rewards])  # shape (N,2)

    # 2) 저장 폴더/파일 경로
    out_dir  = os.path.join(proj_root, 'python', 'data')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'train_history.csv')
    png_path = os.path.join(out_dir, 'train_history.png')

    # 3) CSV 저장
    np.savetxt(
        csv_path, data,
        fmt=['%d','%.6f'],
        delimiter=',',
        header='maskIdx,reward',
        comments=''
    )
    print(f"[INFO] Saved training history CSV to {csv_path}")

    # 4) 그래프 그리기
    steps       = np.arange(len(training_rewards))
    mask_idxs   = data[:,0]
    rewards     = data[:,1]

    plt.figure(figsize=(8,4))
    sc = plt.scatter(steps, rewards, c=mask_idxs, cmap='tab20', s=12, alpha=0.7)
    plt.colorbar(sc, label='Mask Index')
    plt.xlabel('Training Step')
    plt.ylabel('Reward')
    plt.title('DDQN Training: Reward vs Mask Combination')
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    print(f"[INFO] Saved training history plot to {png_path}")

atexit.register(_save_training_history)

# ── 경로 설정 ──────────────────────────────────────────────
script_dir  = os.path.dirname(os.path.abspath(__file__))
proj_root   = os.path.abspath(os.path.join(script_dir, os.pardir))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from models.ddqn.ddqn_agent import QNetwork, update_q_network, ReplayBuffer
from models.ddqn.simulator_env import SimulatorEnv
import metrics

# ── 하이퍼파라미터 ────────────────────────────────────────
PORT          = 5555
HIDDEN_DIM    = 128
BUFFER_SIZE   = 10_000
BATCH_SIZE    = 64
GAMMA         = 0.99
LR            = 1e-3
WEIGHT_DECAY  = 1e-5
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH    = os.path.join(script_dir, "..", "models", "ddqn_ccmask.pth")

# ── 네트워크 초기화 ───────────────────────────────────────
env          = SimulatorEnv()          # state/action 차원 확인용
STATE_DIM    = env.state_size * 2      # pfMetric + prevMask
ACTION_DIM   = env.action_size

q_net      = QNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
target_net = QNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)

if os.path.exists(MODEL_PATH):
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    q_net.load_state_dict(ckpt)
    target_net.load_state_dict(ckpt)
    print(f"Loaded pretrained model: {MODEL_PATH}")
else:
    target_net.load_state_dict(q_net.state_dict())

target_net.eval()
optimizer = optim.Adam(q_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
buffer    = ReplayBuffer(BUFFER_SIZE)

# ── ZeroMQ 세팅 ───────────────────────────────────────────
context = zmq.Context()
socket  = context.socket(zmq.REP)
socket.bind(f"tcp://*:{PORT}")
print(f"DDQN selector server listening on port {PORT} (device={DEVICE})")

poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)

def shutdown(*_):
    print("\nShutting down server …")
    socket.close()
    context.term()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)

# ── 메인 루프 ─────────────────────────────────────────────
while True:
    socks = dict(poller.poll(1000))          # 1 s polling
    if socket in socks and socks[socket] == zmq.POLLIN:
        msg   = socket.recv_json()
        mtype = msg.get("type")

        # ---------- Inference ----------
        if mtype == "infer":
            pf = np.array(msg.get("pfMetric", []),  dtype=np.float32)
            pm = np.array(msg.get("prevMask", []),  dtype=np.float32)
            if pf.size == 0:                   # 필수 필드 체크
                socket.send_json({"error": "missing pfMetric"})
                continue
            if pm.size == 0:
                pm = np.zeros_like(pf)

            # Zero padding to 54
            pad_pf = max(env.state_size - pf.size, 0)
            pad_pm = max(env.state_size - pm.size, 0)
            pf_padded = np.pad(pf, (0, pad_pf), mode='constant')[:env.state_size]
            pm_padded = np.pad(pm, (0, pad_pm), mode='constant')[:env.state_size]
            state_vec = np.concatenate([pf_padded, pm_padded])
            tens      = torch.from_numpy(state_vec).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                q_vals = q_net(tens).cpu().numpy().flatten()

            action = int(np.argmax(q_vals))
            socket.send_json({"maskIdx": action, "qValues": q_vals.tolist()})

        # ---------- Training ----------
        elif mtype == "train":
            exp = msg.get("exp", {})
            try:
                s      = np.array(exp["state"],      dtype=np.float32)
                s_raw  = np.array(exp["state"],      dtype=np.float32)
                # pf, pm 분리
                half   = s_raw.size // 2
                pf_raw = s_raw[:half]
                pm_raw = s_raw[half:]
                # pf_pad 계산
                pad_len_pf = max(env.state_size - pf_raw.size, 0)
                pf_pad = np.pad(pf_raw, (0, pad_len_pf), mode='constant')[:env.state_size]

                # pm_pad 계산
                pad_len_pm = max(env.state_size - pm_raw.size, 0)
                pm_pad = np.pad(pm_raw, (0, pad_len_pm), mode='constant')[:env.state_size]

                # 다시 합쳐서 108차원 벡터로
                s = np.concatenate([pf_pad, pm_pad])
                act_raw = exp["action"]
                if isinstance(act_raw, list):
                    a = int(np.argmax(np.array(act_raw, dtype=np.int8)))
                else:
                    a = int(act_raw)
                r      = float(exp["reward"])
                
                ns_raw = exp["next_state"]
                done   = bool(exp["done"])
                ns     = None if ns_raw is None else np.array(ns_raw, dtype=np.float32)
            except KeyError as e:
                socket.send_json({"error": f"missing key {e}"})
                continue


            # next_state(ns)도 마찬가지로 108차원 맞추기
            if ns is not None:
                ns_raw = np.array(exp["next_state"], dtype=np.float32)
                pf_raw = ns_raw[:half]; pm_raw = ns_raw[half:]
                # pf_pad 계산
                pad_len_pf = max(env.state_size - pf_raw.size, 0)
                pf_pad = np.pad(pf_raw, (0, pad_len_pf), mode='constant')[:env.state_size]

                # pm_pad 계산
                pad_len_pm = max(env.state_size - pm_raw.size, 0)
                pm_pad = np.pad(pm_raw, (0, pad_len_pm), mode='constant')[:env.state_size]
                ns     = np.concatenate([pf_pad, pm_pad])
            buffer.push(s, a, r, ns, done)
            metrics.record_reward(r)
            training_actions.append(a)
            training_rewards.append(r)
            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                update_q_network(
                    q_net, target_net, optimizer, batch,
                    args=type("obj", (object,), {"gamma": GAMMA, "device": DEVICE})
                )
            socket.send_json({"status": "trained"})

        # ---------- Unknown ----------
        else:
            socket.send_json({"error": "unknown message type"})

