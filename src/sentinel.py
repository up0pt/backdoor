import torch
import torch.nn as nn
import copy
import math

def cosine_similarity(params_p: dict, params_m: dict, eps: float = 1e-8) -> float:
    """
    Algorithm 2: Cosine Similarity
    ・params_p, params_m はモデルの state_dict 相当の dict
    ・全レイヤーのコサイン類似度を平均して返す
    """
    sim_sum = 0.0
    n_layers = 0
    for key in params_m:
        v_m = params_m[key].flatten()
        v_p = params_p[key].flatten()
        # 内積 / (||v_m|| * ||v_p||)
        cos = torch.dot(v_m, v_p) / (v_m.norm() * v_p.norm() + eps)
        sim_sum += cos.item()
        n_layers += 1
    return sim_sum / n_layers if n_layers > 0 else sim_sum


def compute_bootstrap_loss(model: nn.Module,
                           bootstrap_loader: torch.utils.data.DataLoader,
                           loss_fn: nn.Module,
                           device: torch.device = torch.device('cpu')) -> float:
    """
    Algorithm 3: Compute Bootstrap Loss
    ・モデルにパラメータを読み込んだ上で、D_bs 上の平均損失を返す
    """
    model = copy.deepcopy(model).to(device).eval()
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for x, y in bootstrap_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            batch_loss = loss_fn(y_pred, y).item() * x.size(0)
            total_loss += batch_loss
            n_samples += x.size(0)
    return total_loss / max(n_samples, 1)


def map_loss_distance(history_i: list,
                      history_j: list,
                      tau_L: float,
                      l_min: float = 1e-3) -> float:
    """
    Algorithm 4: Map Loss Distance
    ・history_i, history_j は各ラウンドの損失を格納したリスト
    """
    # φ は履歴の平均
    bar_li = sum(history_i) / len(history_i)
    bar_lj = sum(history_j) / len(history_j)
    # κ = 1 / max(bar_li, l_min)
    kappa = 1.0 / max(bar_li, l_min)
    # Δl = max(bar_lj - bar_li, 0)
    delta = max(bar_lj - bar_li, 0.0)
    w = math.exp(-kappa * delta)
    return w if w >= tau_L else 0.0


def normalize_model(params_p: dict,
                    params_m: dict) -> dict:
    """
    Algorithm 5: Normalize Model
    ・各レイヤーごとに ‖P[l]‖ / ‖M[l]‖ を上限 1.0 でクリップし、
      P[l] に掛け合わせる
    """
    normalized = {}
    for key in params_m:
        norm_m = params_m[key].norm().item()
        norm_p = params_p[key].norm().item()
        rho = min(1.0, norm_p / (norm_m + 1e-8))
        normalized[key] = params_p[key] * rho
    return normalized


def sentinel_aggregation(
                         client_id: int,
                         neighbor_params: dict,
                         local_model: nn.Module,
                         loss_history: dict,
                         bootstrap_loader: torch.utils.data.DataLoader,
                         loss_fn: nn.Module,
                         tau_S: float,
                         tau_L: float,
                         l_min: float = 1e-3,
                         device: torch.device = torch.device('cpu')) -> dict:
    """
    Algorithm 1: Sentinel Aggregation
    ・neighbor_params: { client_id: state_dict }（自身 i も含む想定）
    ・loss_history:   { client_id: [loss_round0, loss_round1, ...] }
    ・bootstrap_loader, loss_fn: ブートストラップ検証用
    ・tau_S: 類似度フィルタ閾値, tau_L: 重み w の閾値
    """
    i = client_id

    # local model のパラメータ
    M_dict = local_model.state_dict()
    M_dict = {k: v.to(device) for k, v in M_dict.items()}

    # (1) Similarity filtering
    filtered = {}
    for j, Pj in neighbor_params.items():
        if j == i:
            continue
        # 隣接モデルのパラメータも同じ device に移動
        Pj = {k: v.to(device) for k, v in Pj.items()}
        S_j = cosine_similarity(Pj, M_dict)
        if S_j >= tau_S:
            filtered[j] = Pj

    # (1.5) 自身の履歴に現在のモデル損失を追加
    L_i = compute_bootstrap_loss(local_model, bootstrap_loader, loss_fn, device)
    loss_history[i].append(L_i)

    # (2) Bootstrap validation & loss history 更新 & 重み計算
    weights = {}
    for j, Pj in filtered.items():
        # モデルにパラメータ読み込み
        temp_model = copy.deepcopy(local_model).to(device)
        temp_model.load_state_dict(Pj)
        L_j = compute_bootstrap_loss(temp_model, bootstrap_loader, loss_fn, device)
        loss_history.setdefault(j, []).append(L_j)
        w_j = map_loss_distance(loss_history[i], loss_history[j], tau_L, l_min)
        if w_j > 0:
            weights[j] = w_j
    
    # (3) Layer normalization ＆ 重み付き平均
    # 正規化済み P̃ の準備
    normalized_P = {
        j: normalize_model(Pj, M_dict)
        for j, Pj in filtered.items()
        if j in weights
    }

    total_w = sum(weights.values())
    if total_w == 0:
        # 全ての重みがゼロならローカルモデルを返す
        return M_dict

    # aggregate into θ
    theta = {k: torch.zeros_like(v) for k, v in M_dict.items()}
    for j, Ptil in normalized_P.items():
        wj = weights[j]
        for k in theta:
            theta[k] += wj * Ptil[k]
    # 正規化
    for k in theta:
        theta[k] /= total_w

    return theta
