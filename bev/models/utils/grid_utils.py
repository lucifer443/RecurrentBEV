import torch

def gen_dx_bx(xbound, ybound, zbound):
    # 每个BEV像素的宽度
    dx = torch.Tensor([
        row[2]
        for row in [xbound, ybound, zbound]])
    # 第一个BEV像素中心点的坐标
    bx = torch.Tensor([
        row[0] + row[2] / 2.0
        for row in [xbound, ybound, zbound]])
    # BEV像素个数
    # BaseModule会对model的每个parameter做mean()，需要每个parameter都是float类型
    nx = torch.Tensor([
        round((row[1] - row[0]) / row[2])
        for row in [xbound, ybound, zbound]])

    return dx, bx, nx
