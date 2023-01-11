import torch

T = torch.Tensor


def ref_fw_simple(log_probs: T, ys: T):
    assert log_probs.dim() == 3
    assert ys.dim() == 1
    assert log_probs.size(1) == ys.size(0) + 1

    T, U = log_probs.shape[:2]
    U -= 1

    # init with all 1.0 to debug
    alphas = torch.full((T+1, 2*U+1), fill_value=1.0, device=log_probs.device)
    alphas[0, 0] = 0.0
    for t in range(1, T):
        alphas[t, 0] = alphas[t-1, 0] + log_probs[t-1, 0, 0]

    alphas[0, 1:].fill_(float('-inf'))

    for s in range(1, alphas.size(1)):
        for t in range(1, T+1):
            if t == T and s < alphas.size(1)-2:
                continue
            u = (s+1)//2
            yn = 0 if (s % 2 == 0) else ys[u-1]
            alphas[t, s] = torch.logaddexp(
                alphas[t-1, s] + log_probs[t-1, u, yn],
                alphas[t-1, s-1] + log_probs[t-1, s//2, yn]
            )
            if s > 1 and s % 2 == 1 and yn != ys[u-2]:
                alphas[t, s] = torch.logaddexp(
                    alphas[t, s],
                    alphas[t-1, s-2] + log_probs[t-1, u-1, yn]
                )
    return alphas, torch.logaddexp(alphas[T, 2*U], alphas[T, 2*U-1])


def ref_bw_simple(log_probs: T, ys: T):
    assert log_probs.dim() == 3
    assert ys.dim() == 1
    assert log_probs.size(1) == ys.size(0) + 1

    T, U = log_probs.shape[:2]
    U -= 1

    # init with all 1.0 to debug
    betas = torch.full((T+1, 2*U+1), fill_value=1.0, device=log_probs.device)
    T1 = T
    S1 = 2*U
    betas[T1, S1] = 0.0
    betas[T1, S1-1] = 0.0
    betas[T1, :-2].fill_(float('-inf'))

    for t in range(1, T):
        betas[T1-t, S1] = betas[T1-t+1, S1] + log_probs[T1-t, U, 0]

    for s in range(1, betas.size(1)):
        for t in range(1, T+1):
            u = (S1-s+1)//2
            isreal = (s % 2 == 1)
            betas[T1-t, S1-s] = torch.logaddexp(
                betas[T1-t+1, S1-s] +
                log_probs[T1-t, u, (ys[u-1] if isreal else 0)],
                betas[T1-t+1, S1-s+1] +
                log_probs[T1-t, u, (0 if isreal else ys[u])]
            )
            if s > 1 and isreal and ys[u-1] != ys[u]:
                betas[T1-t, S1-s] = torch.logaddexp(
                    betas[T1-t, S1-s],
                    betas[T1-t+1, S1-s+2] + log_probs[T1-t, u, ys[u]]
                )
    return betas, betas[0, 0]


def ref_fw_shift(log_probs: T, ys: T):
    assert log_probs.dim() == 3
    assert ys.dim() == 1
    assert log_probs.size(1) == ys.size(0) + 1

    T, U = log_probs.shape[:2]
    U -= 1

    sT = T-U+1
    sU = 2*U+1
    alphas = torch.full((sT, sU), fill_value=1.0, device=log_probs.device)
    alphas[0, 0] = 0.0
    # init s = 0
    for i in range(1, sT):
        alphas[i, 0] = alphas[i-1, 0] + log_probs[i-1, 0, 0]
    alphas[0, 1] = log_probs[0, 0, ys[0]]
    # init i = 0
    for s in range(2, sU):
        shift = s//2 + 1
        if shift - 1 >= T:
            continue
        u = (s-1)//2
        if s % 2 == 0:
            alphas[0, s] = alphas[0, s-1] + log_probs[shift-1, s//2, 0]
        elif ys[u] != ys[u-1]:
            alphas[0, s] = alphas[0, s-2] + log_probs[shift-1, (s-1)//2, ys[u]]
        else:
            alphas[0, s] = float('-inf')

    # init s = 1, ts = 1, u = 0
    for i in range(1, sT):
        alphas[i, 1] = torch.logaddexp(
            alphas[i-1, 1] + log_probs[i, 1, ys[0]],
            alphas[i, 0] + log_probs[i, 0, ys[0]]
        )

    for s in range(2, sU):
        ts = s//2 + 1
        u = (s-1)//2
        for i in range(1, sT):
            if i + ts - 1 >= T:
                continue

            if s % 2 == 0:
                alphas[i, s] = torch.logaddexp(
                    alphas[i-1, s] + log_probs[i+ts-1, (s+1)//2, 0],
                    alphas[i, s-1] + log_probs[i+ts-1, s//2, 0]
                )
            else:
                alphas[i, s] = torch.logaddexp(
                    alphas[i-1, s] + log_probs[i+ts-1, (s+1)//2, ys[u]],
                    alphas[i-1, s-1] + log_probs[i+ts-1, s//2, ys[u]]
                )
                if s > 1 and ys[u] != ys[u-1]:
                    alphas[i, s] = torch.logaddexp(
                        alphas[i, s],
                        alphas[i, s-2] + log_probs[i+ts-1, (s-1)//2, ys[u]]
                    )

    costs = alphas[sT-1, sU-2]
    if sT > 1:
        costs = torch.logaddexp(alphas[sT-2, sU-1], costs)
    return alphas, costs


def ref_bw_shift(log_probs: T, ys: T):
    assert log_probs.dim() == 3
    assert ys.dim() == 1
    assert log_probs.size(1) == ys.size(0) + 1

    T, U = log_probs.shape[:2]
    U -= 1

    sT = T-U+1
    sU = 2*U+1
    betas = torch.full((sT, sU), fill_value=1.0, device=log_probs.device)
    sT1 = sT-1
    sU1 = sU-1

    betas[sT1-1, sU1] = 0.0
    betas[sT1, sU1-1] = 0.0
    # init last row of betas
    # t_min = s//2 + 1 = U + 1
    # t_max = T - (2*U-sU1)//2 = T
    # t=T has been initialized, so here we start from T-1
    for t in range(T-1, U, -1):
        i = t - (U+1)
        betas[i, sU1] = betas[i+1, sU1] + log_probs[t, U, 0]
    # init last column of betas
    for s in range(sU1-2, 0, -1):
        u = (s-1)//2
        if (s % 2 == 1) and ys[u] != ys[u+1]:
            betas[sT1, s] = betas[sT1, s+2] + \
                log_probs[sT1+s//2+1, s//2+1, ys[u+1]]
        else:
            betas[sT1, s] = float('-inf')

    # calculate rest of the betas
    for s in range(sU1-1, 0, -1):
        for i in range(sT1-1, -1, -1):
            ts = s//2 + 1
            u = (s-1)//2
            if s % 2 == 0:
                betas[i, s] = torch.logaddexp(
                    betas[i+1, s] + log_probs[i+ts, u+1, 0],
                    betas[i+1, s+1] + log_probs[i+ts, u+1, ys[u+1]]
                )
            else:
                betas[i, s] = torch.logaddexp(
                    betas[i+1, s] + log_probs[i+ts, u+1, ys[u]],
                    betas[i, s+1] + log_probs[i+ts, u+1, 0]
                )
                if s < sU1 - 1 and ys[u] != ys[u+1]:
                    betas[i, s] = torch.logaddexp(
                        # clone for not modifying tensor in-place
                        betas[i, s].clone(),
                        betas[i, s+2] + log_probs[i+ts, u+1, ys[u+1]]
                    )
    # special dealing s = 0, ts = 0, u = -1
    # t_min = ts = 0, t_max = T - (2U-s)//2 = T-U, i = t-ts = t
    betas[T-U, 0] = betas[T-U, 1] + log_probs[T-U, 0, ys[0]]
    for i in range(T-U-1, -1, -1):
        betas[i, 0] = torch.logaddexp(
            betas[i+1, 0] + log_probs[i, 0, 0],
            betas[i, 1] + log_probs[i, 0, ys[0]]
        )
    return betas, betas[0, 0]


def collect_units(log_probs: T, ys: T):
    N, T, U = log_probs.shape[:3]
    index = torch.full((N, U, 3), fill_value=0,
                       device=log_probs.device, dtype=torch.long)
    index[:, 1:, 1] = ys
    index[:, :U-1, 2] = ys
    index = index.unsqueeze(1).expand(-1, T, -1, -1)
    return log_probs.gather(dim=-1, index=index)


if __name__ == "__main__":
    torch.manual_seed(1)

    T, U, V = 5, 2, 4
    log_probs = torch.randn(T, U+1, V).log_softmax(dim=-1)
    ys = torch.randint(1, V, (U, ))
    print(ys.tolist())

    alphas, a_ll = ref_fw_simple(log_probs, ys)
    print(f"alpha cost: {-a_ll.item():.4f}")
    print(alphas.T.flip(0))

    s_alphas, sa_ll = ref_fw_shift(log_probs, ys)
    print(f"shift alpha cost: {-sa_ll.item():.4f}")
    print(s_alphas.T.flip(0))

    betas, b_ll = ref_bw_simple(log_probs, ys)
    print(f"beta cost: {-b_ll.item():.4f}")
    print(betas.T.flip(0))

    s_betas, sb_ll = ref_bw_shift(log_probs, ys)
    print(f"shift beta cost: {-sb_ll.item():.4f}")
    print(s_betas.T.flip(0))
