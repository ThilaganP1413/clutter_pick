"""M7 piece 2: dilation exactness, shapes, uniform init, masked sampling, gradients, timing/memory."""
import argparse
import math
import time

import torch
import torchvision


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()
    from cpick.models.picknet import GRID, PickNet, masked_dist, resnet34_8s

    torch.manual_seed(0)
    dev, B = args.device, args.batch
    ok = True

    # 1) Dilation exactness: patched layers, subsampled at the old stride == stock strided layers.
    ref = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1).eval()
    dil = resnet34_8s(pretrained=True).eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        h = ref.layer2(ref.layer1(ref.maxpool(ref.relu(ref.bn1(ref.conv1(x))))))
        r3 = ref.layer3(h)
        r4 = ref.layer4(r3)
        d3 = dil.layer3(h)
        d4 = dil.layer4(d3)
    e3 = ((d3[:, :, ::2, ::2] - r3).abs().max() / r3.abs().max()).item()
    e4 = ((d4[:, :, ::4, ::4] - r4).abs().max() / r4.abs().max()).item()
    print(f"[1] layer3 {tuple(d3.shape)} rel err {e3:.1e}; layer4 {tuple(d4.shape)} rel err {e4:.1e}")
    ok &= d4.shape[-1] == 32 and e3 < 1e-4 and e4 < 1e-4

    # 2) Shapes and exactly-uniform initial policy inside the mask.
    net = PickNet().to(dev)
    depth = 0.74 - 0.1 * torch.rand(B, 256, 256, device=dev)
    mask = torch.zeros(B, GRID, GRID, dtype=torch.bool, device=dev)
    mask[:, 30:90, 25:105] = True
    logits, value = net(depth)
    dist = masked_dist(logits, mask)
    ent, uni = dist.entropy().mean().item(), math.log(mask[0].sum().item())
    print(f"[2] logits {tuple(logits.shape)}, value {tuple(value.shape)}; init entropy {ent:.4f} vs log(#valid) {uni:.4f}")
    ok &= logits.shape == (B, GRID, GRID) and value.shape == (B,) and abs(ent - uni) < 1e-3

    # 3) Sampling never leaves the mask, also for sharp random logits.
    sharp = masked_dist(10 * torch.randn_like(logits), mask)
    inside = all(mask.flatten(1).gather(1, d.sample((1000,)).T).all().item() for d in (dist, sharp))
    print(f"[3] 2x1000 samples per item inside mask: {inside}; sharp entropy finite: {torch.isfinite(sharp.entropy()).all().item()}")
    ok &= inside

    # 4) Gradients reach the head and the encoder.
    a = dist.sample()
    loss = -dist.log_prob(a).mean() + value.pow(2).mean()
    loss.backward()
    g_head, g_enc = net.head.weight.grad.norm().item(), net.layer1[0].conv1.weight.grad.norm().item()
    print(f"[4] grad norm head {g_head:.2e}, encoder layer1 {g_enc:.2e}")
    ok &= g_head > 0 and g_enc > 0

    # 5) Timing / memory: one training step (fwd+bwd) at batch B, and a batch-1 inference.
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    sync = (lambda: torch.cuda.synchronize()) if dev.startswith("cuda") else (lambda: None)
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for k in range(13):
        if k == 3:
            sync(); t0 = time.perf_counter()
        lg, v = net(depth)
        l = -masked_dist(lg, mask).log_prob(a).mean() + v.pow(2).mean()
        opt.zero_grad(); l.backward(); opt.step()
    sync(); t_train = (time.perf_counter() - t0) / 10
    with torch.no_grad():
        for k in range(13):
            if k == 3:
                sync(); t0 = time.perf_counter()
            net(depth[:1])
        sync(); t_inf = (time.perf_counter() - t0) / 10
    n_par = sum(q.numel() for q in net.parameters()) / 1e6
    mem = f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if dev.startswith("cuda") else "n/a"
    sps = 1.0 / (t_inf + 3 * t_train / B)        # 1 inference per env step + 3 PPO epochs of training
    print(f"[5] {n_par:.1f} M params; train step (B={B}) {t_train * 1e3:.0f} ms, inference (B=1) {t_inf * 1e3:.1f} ms, "
          f"peak mem {mem}; network-limited env steps/s ~{sps:.0f}")

    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED -- paste this output")


if __name__ == "__main__":
    main()