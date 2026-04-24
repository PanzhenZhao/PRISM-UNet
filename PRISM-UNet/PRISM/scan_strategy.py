import torch

_PERM_CACHE = {}


def _get_scan_perm(H: int, W: int, device: torch.device, scan_type: str):
    
    key = (H, W, str(device), scan_type)
    if key in _PERM_CACHE:
        return _PERM_CACHE[key]

    if scan_type == 'scan3':
        perm, inv_perm = _build_scan3_perm(H, W, device)
    elif scan_type == 'scan4':
        perm, inv_perm = _build_scan4_perm(H, W, device)
    else:
        raise ValueError(f"Unknown scan_type: {scan_type}")

    _PERM_CACHE[key] = (perm, inv_perm)
    return perm, inv_perm


def _build_scan3_perm(H: int, W: int, device):
    order = []
    for d in range(W - 1, -H, -1):
        h_min = max(0, -d)
        h_max = min(H - 1, W - 1 - d)
        hs = list(range(h_min, h_max + 1))
        if d % 2 == 0:
            hs = hs[::-1]
        for h in hs:
            w = h + d
            order.append(h * W + w)

    perm = torch.tensor(order, dtype=torch.long, device=device)
    L = H * W
    inv_perm = torch.empty(L, dtype=torch.long, device=device)
    inv_perm[perm] = torch.arange(L, device=device)
    return perm, inv_perm


def _build_scan4_perm(H: int, W: int, device):
    order = []
    top, bottom = 0, H - 1
    left, right = 0, W - 1

    while top <= bottom and left <= right:
        for h in range(bottom, top - 1, -1):
            order.append(h * W + left)
        left += 1
        if left > right: break
        for w in range(left, right + 1):
            order.append(top * W + w)
        top += 1
        if top > bottom: break
        for h in range(top, bottom + 1):
            order.append(h * W + right)
        right -= 1
        if left > right: break
        for w in range(right, left - 1, -1):
            order.append(bottom * W + w)
        bottom -= 1

    perm = torch.tensor(order, dtype=torch.long, device=device)
    L = H * W
    inv_perm = torch.empty(L, dtype=torch.long, device=device)
    inv_perm[perm] = torch.arange(L, device=device)
    return perm, inv_perm


# ==========================================
# parallel scan 1
# ==========================================
class CrossScan_1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 1, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        y = ys[:, 0]
        return y.view(B, -1, H, W)


class CrossMerge_1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W, K)  # 优化点 2：显式保存真实的 K 值
        ys = ys.view(B, K, D, -1)
        y = ys[:, 0]
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W, K = ctx.shape
        B, C, L = x.shape
        xs = x.new_zeros((B, K, C, L))  # 优化点 2：使用 new_zeros 防治未使用的 K 维度产生脏梯度
        xs[:, 0] = x
        return xs.view(B, K, C, H, W)


# ==========================================
# vertical scan 2 (Reversed / 逆序)
# ==========================================
class CrossScan_2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 1, C, H * W))
        xs[:, 0] = x.transpose(dim0=2, dim1=3).flatten(2, 3).flip(dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        y = ys[:, 0].flip(dims=[-1]).view(B, C, W, H).transpose(dim0=2, dim1=3).contiguous()
        return y


class CrossMerge_2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W, K)
        ys = ys.view(B, K, D, -1)
        # 优化点 3：末尾加上 flatten(2, 3) 强制将 4D 张量铺平为 3D (B, D, L)，消除 Shape Bug
        y = ys[:, 0].flip(dims=[-1]).view(B, D, W, H).transpose(dim0=2, dim1=3).flatten(2, 3)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W, K = ctx.shape
        B, C, L = x.shape
        xs = x.new_zeros((B, K, C, L))
        xs[:, 0] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3).flip(dims=[-1])
        return xs.view(B, K, C, H, W)


# ==========================================
# scan 3 (Diagonal)
# ==========================================
class CrossScan_3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        perm, inv_perm = _get_scan_perm(H, W, x.device, 'scan3')  # 应用全局缓存
        ctx.save_for_backward(inv_perm)

        xs = x.new_empty((B, 1, C, H * W))
        x_flat = x.flatten(2, 3)
        xs[:, 0] = x_flat.index_select(-1, perm)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        (inv_perm,) = ctx.saved_tensors
        y = ys[:, 0]
        y = y.index_select(-1, inv_perm)
        return y.view(B, -1, H, W)


class CrossMerge_3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W, K)
        perm, inv_perm = _get_scan_perm(H, W, ys.device, 'scan3')
        ctx.save_for_backward(perm)

        ys = ys.view(B, K, D, -1)
        seq = ys[:, 0]
        y = seq.index_select(-1, inv_perm)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W, K = ctx.shape
        (perm,) = ctx.saved_tensors
        B, C, L = x.shape
        xs = x.new_zeros((B, K, C, L))
        xs[:, 0] = x.index_select(-1, perm)
        return xs.view(B, K, C, H, W)


# ==========================================
# scan 4 (Spiral)
# ==========================================
class CrossScan_4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        perm, inv_perm = _get_scan_perm(H, W, x.device, 'scan4')
        ctx.save_for_backward(inv_perm)

        xs = x.new_empty((B, 1, C, H * W))
        x_flat = x.flatten(2, 3)
        xs[:, 0] = x_flat.index_select(-1, perm)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        (inv_perm,) = ctx.saved_tensors
        y = ys[:, 0]
        y = y.index_select(-1, inv_perm)
        return y.view(B, -1, H, W)


class CrossMerge_4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W, K)
        perm, inv_perm = _get_scan_perm(H, W, ys.device, 'scan4')
        ctx.save_for_backward(perm)

        ys = ys.view(B, K, D, -1)
        seq = ys[:, 0]
        y = seq.index_select(-1, inv_perm)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W, K = ctx.shape
        (perm,) = ctx.saved_tensors
        B, C, L = x.shape
        xs = x.new_zeros((B, K, C, L))
        xs[:, 0] = x.index_select(-1, perm)
        return xs.view(B, K, C, H, W)