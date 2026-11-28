"""Minimal PPO for one-step episodes over a masked 128x128 categorical. Own code, CODs-style settings."""
from collections import defaultdict

import torch
import torch.nn.functional as F

from cpick.models.picknet import masked_dist


class Rollout:
    """Fixed-size buffer on the training device. The mask is stored and reused in the update."""

    def __init__(self, n, h, w, grid, device):
        self.n, self.dev = n, device
        self.depth = torch.zeros(n, h, w, device=device)
        self.mask = torch.zeros(n, grid, grid, dtype=torch.bool, device=device)
        self.action = torch.zeros(n, dtype=torch.long, device=device)
        self.logp = torch.zeros(n, device=device)
        self.value = torch.zeros(n, device=device)
        self.reward = torch.zeros(n, device=device)
        self.i = 0

    def reset(self):
        self.i = 0

    def add(self, depth, mask, action, logp, value, reward):
        k = self.i
        self.depth[k], self.mask[k], self.action[k] = depth, mask, action
        self.logp[k], self.value[k], self.reward[k] = logp, value, reward
        self.i += 1


@torch.no_grad()
def act(net, depth_np, mask_np, device, greedy=False):
    """One observation -> (action, log-prob, value, depth tensor, mask tensor)."""
    depth = torch.as_tensor(depth_np, device=device)
    mask = torch.as_tensor(mask_np, device=device).bool()
    logits, v_logit = net(depth[None])
    d = masked_dist(logits, mask[None])
    a = d.logits.argmax(-1) if greedy else d.sample()
    return int(a), float(d.log_prob(a)), float(torch.sigmoid(v_logit)), depth, mask


def ppo_update(net, opt, buf, rl):
    adv = buf.reward - buf.value          # rewards fixed in {0, 1}: raw advantages are already well scaled
    s, first, stop = defaultdict(list), True, False
    for _ in range(rl["epochs"]):
        for idx in torch.randperm(buf.n, device=buf.dev).split(rl["minibatch"]):
            logits, v_logit = net(buf.depth[idx])
            d = masked_dist(logits, buf.mask[idx])
            logratio = d.log_prob(buf.action[idx]) - buf.logp[idx]
            ratio = logratio.exp()
            kl = ((ratio - 1) - logratio).mean().item()
            if first:                                    # before any step: ratio must equal 1, KL = 0
                s["first_ratio_dev"].append((ratio - 1).abs().max().item())
                first = False
            else:
                if rl.get("target_kl") and kl > rl["target_kl"]:
                    s["stop_kl"].append(kl)              # the value that ended this update
                    stop = True
                    break
                s["kl"].append(kl)
            a = adv[idx]
            pg = -torch.min(ratio * a, ratio.clamp(1 - rl["clip"], 1 + rl["clip"]) * a).mean()
            vf = F.binary_cross_entropy_with_logits(v_logit, buf.reward[idx])
            ent = d.entropy().mean()
            loss = pg + rl["vf_coef"] * vf - rl["ent_coef"] * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), rl["max_grad_norm"])
            opt.step()
            s["pg"].append(pg.item()); s["vf"].append(vf.item()); s["ent"].append(ent.item())
            s["grad_norm"].append(gn.item())
            s["clipfrac"].append(((ratio - 1).abs() > rl["clip"]).float().mean().item())
        if stop:
            break
    out = {k: sum(v) / len(v) for k, v in s.items() if v}
    out["minibatches"] = len(s["pg"])                   # 12 = full update; fewer = KL limit hit
    return out