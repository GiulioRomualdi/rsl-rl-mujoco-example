import os
import torch
import pytest

from RL.modules.actor_critic_moe import ActorCriticMOE

@pytest.fixture(scope="module")
def moe_model():
    cfg = {
        "env": {"obs_dim": 276, "action_dim": 31},
        "policy": {
            "num_experts": 2,
            "expert_path": "model/test_moe",  # 替换成你的实际路径
            "moe_hidden_dims": [64, 32],
            "moe_activation": "relu",
            "actor_hidden_dims": [1024, 512, 512, 256],
            "critic_hidden_dims": [1024, 512, 512, 256],
            "init_noise_std": 0.1,
            "noise_std_type": "scalar",
        },
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ActorCriticMOE(
        num_actor_obs=cfg["env"]["obs_dim"],
        num_critic_obs=cfg["env"]["obs_dim"],
        num_actions=cfg["env"]["action_dim"],
        num_experts=cfg["policy"]["num_experts"],
        expert_path=cfg["policy"]["expert_path"],
        moe_hidden_dims=cfg["policy"]["moe_hidden_dims"],
        moe_activation=cfg["policy"]["moe_activation"],
        actor_hidden_dims=cfg["policy"]["actor_hidden_dims"],
        critic_hidden_dims=cfg["policy"]["critic_hidden_dims"],
        init_noise_std=cfg["policy"]["init_noise_std"],
        noise_std_type=cfg["policy"]["noise_std_type"],
        device=device,
    ).to(device)
    return model, device, cfg

def test_act_and_shapes(moe_model):
    model, device, cfg = moe_model
    B = 8
    obs = torch.randn(B, cfg["env"]["obs_dim"], device=device)

    # 1) act() 输出 shape
    actions = model.act(obs)
    assert actions.shape == (B, cfg["env"]["action_dim"])

    # 2) act_inference() 返回 shape
    means = model.act_inference(obs)
    assert means.shape == (B, cfg["env"]["action_dim"])

def test_log_prob_and_finiteness(moe_model):
    model, device, cfg = moe_model
    B = 8
    obs = torch.randn(B, cfg["env"]["obs_dim"], device=device)
    actions = model.act(obs)

    # 3) get_actions_log_prob() 输出
    logp = model.get_actions_log_prob(actions)
    assert logp.shape == (B,)
    assert torch.isfinite(logp).all()

def test_evaluate(moe_model):
    model, device, cfg = moe_model
    B = 8
    obs = torch.randn(B, cfg["env"]["obs_dim"], device=device)
    values = model.evaluate(obs)
    assert values.shape == (B, 1)

def test_gate_probabilities(moe_model):
    model, device, cfg = moe_model
    B = 10
    obs = torch.randn(B, cfg["env"]["obs_dim"], device=device)

    with torch.no_grad():
        normed = model.gating_norm(obs)
        probs = model.gate(normed)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

def test_experts_frozen(moe_model):
    model, _, _ = moe_model
    for i, mod in enumerate(model.experts.experts):
        # mod 是 [norm, ActorCritic]
        ac = mod[1]
        for p in ac.parameters():
            assert not p.requires_grad, f"Expert {i} 参数未被冻结"

def test_reproducible_gate_sampling(moe_model):
    model, device, cfg = moe_model
    B = 5
    obs = torch.randn(B, cfg["env"]["obs_dim"], device=device)

    torch.manual_seed(123)
    model.update_distribution(obs)
    idx1 = model._selected_expert_idx.clone()

    torch.manual_seed(123)
    model.update_distribution(obs)
    idx2 = model._selected_expert_idx.clone()

    assert torch.equal(idx1, idx2)

if __name__ == "__main__":
    # 直接运行脚本
    pytest.main([os.path.abspath(__file__)])
