#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
open3d_downsample.py —— 3D 点云下采样五种方法（Open3D 单文件版）

包含
  1. Voxel Grid Downsampling      体素栅格下采样（o3d voxel_down_sample，输出体素质心）
  2. Octree Downsampling          八叉树下采样（自实现递归八分，输出叶子体素中心，行为对齐 pcl::octree）
  3. Random Sample                随机下采样（o3d random_down_sample）
  4. Keypoint (ISS)               关键点下采样（特征保持型，o3d geometry.keypoint.compute_iss_keypoints）
  5. FPS                         最远点采样（o3d farthest_point_down_sample）

输出
  · 每种方法的点数、耗时、压缩比
  · *_orig/voxel/octree/random/keypoint/fps.pcd
  · 弹出对比窗口（2 行 3 列并排显示，左侧文字图例标明「颜色 → 方法」，--no-show 关闭）
  · matplotlib 2x3 对比图（headless 环境同样出图）

用法
  python open3d_downsample.py                              # 读取 data/table_scene_mug_stereo_textured.pcd
  python open3d_downsample.py -i bunny.pcd --voxel 0.004   # 读取 data/bunny.pcd
  python open3d_downsample.py --target 8000 --no-show
  python open3d_downsample.py --iss-salient 0.02 --iss-nms 0.015

数据准备
  把需要的 pcd 文件放到本脚本同级的 data/ 目录下（脚本只在本地读取，不联网下载）
"""
import argparse
import os
import sys
import time

import numpy as np
import open3d as o3d

# ----------------------------------------------------------------------------- 常量
# 点云数据目录：脚本所在目录下的 data/，与运行时的工作目录无关
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_PCD = "table_scene_mug_stereo_textured.pcd"

# ----------------------------------------------------------------------------- 工具
def resolve_pcd(path: str) -> str:
    """把 -i 传入的文件名解析为本地文件：优先 data/ 目录，其次当前工作目录。"""
    candidates = [path] if os.path.isabs(path) else [os.path.join(DATA_DIR, path), path]
    for cand in candidates:
        if os.path.isfile(cand):
            return os.path.abspath(cand)

    hint = ""
    if os.path.isdir(DATA_DIR):
        names = sorted(f for f in os.listdir(DATA_DIR)
                       if f.lower().endswith((".pcd", ".ply")))
        if names:
            hint = "\n        data/ 下现有文件：" + ", ".join(names)
    sys.exit(f"[ERROR] 找不到点云文件 {path}，请把文件放到 {DATA_DIR}/ 下"
             f"或用 -i 指定已存在的文件。{hint}")


def load_cloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        sys.exit(f"[ERROR] 读取失败或点云为空：{path}")
    pts = np.asarray(pcd.points)
    bad = ~np.isfinite(pts).all(axis=1)
    if bad.any():
        print(f"[INFO] 剔除 {int(bad.sum())} 个无效点(NaN/Inf)，有效点 {len(pts) - int(bad.sum())}")
        pcd = pcd.select_by_index(np.where(~bad)[0])
    return pcd


# ------------------------------------------------ 1. Voxel Grid Downsampling
def voxel_downsample(pcd, voxel_size):
    """体素栅格下采样：划分立方体体素，每个体素内所有点的质心作为一个输出点。
    优点：速度快、分布均匀、天然去除噪点；缺点：体素质心会偏离真实表面（"缩水"）。"""
    t0 = time.time()
    out = pcd.voxel_down_sample(voxel_size=voxel_size)
    return out, time.time() - t0


# ------------------------------------------------ 2. Octree Downsampling
def octree_downsample(pcd, resolution=None, max_depth=None, mode="center"):
    """八叉树下采样（自实现，语义对齐 pcl::octree::OctreePointCloud）。

    从包住整朵点云的立方体开始，递归把立方体八等分，直到边长 <= resolution（或达到 max_depth）。
    每个叶子节点只输出 1 个点：
      mode="center"  -> 叶子体素中心坐标（PCL octree 的 getVoxelCenterCoordinate 行为）
      mode="centroid"-> 叶子内点的质心（与 VoxelGrid 一致，便于对照两者差异）
    与 VoxelGrid 的差别：VoxelGrid 的格子原点固定在世界坐标系零点、每个轴独立分格；
    八叉树从点云自身包围盒开始二分，格子会随数据自适应，边界处分组不同 -> 结果不完全相同。
    """
    t0 = time.time()
    pts = np.asarray(pcd.points)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    side = float((hi - lo).max())
    if side <= 0:
        return pcd, time.time() - t0

    if max_depth is None:
        if resolution is None:
            resolution = side / 64.0
        # PCL octree 的做法：depth = ceil(log2(maxExtent / resolution))，
        # 再把包围盒撑成 2^depth * resolution 的立方体（居中于原包围盒中心），
        # 这样叶子体素边长恰好等于 resolution，可与 VoxelGrid 公平对比
        max_depth = max(1, int(np.ceil(np.log2(side / resolution)))) if resolution < side else 0
        side = resolution * (2 ** max_depth)

    c = (lo + hi) * 0.5
    cube = (c - side * 0.5, c + side * 0.5)

    out_pts = []
    stack = [(pts, cube, 0)]
    while stack:
        p, (cb0, cb1), depth = stack.pop()
        if depth >= max_depth or len(p) <= 1:
            if mode == "centroid":
                out_pts.append(p.mean(axis=0))
            else:
                out_pts.append((cb0 + cb1) * 0.5)
            continue
        mid = (cb0 + cb1) * 0.5
        # 3 位编码：bit2=x, bit1=y, bit0=z，大于等于中位面的点归入高位
        idx = (((p[:, 0] >= mid[0]).astype(np.int8) << 2)
               | ((p[:, 1] >= mid[1]).astype(np.int8) << 1)
               | (p[:, 2] >= mid[2]).astype(np.int8))
        for code in range(8):
            sel = idx == code
            if not np.any(sel):
                continue
            lo2 = np.array([mid[i] if (code >> (2 - i)) & 1 else cb0[i] for i in range(3)])
            hi2 = np.array([cb1[i] if (code >> (2 - i)) & 1 else mid[i] for i in range(3)])
            stack.append((p[sel], (lo2, hi2), depth + 1))

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(np.asarray(out_pts))
    return out, time.time() - t0


# ------------------------------------------------ 3. Random Sample
def random_downsample(pcd, n):
    """随机下采样：从原云中均匀随机抽取 n 个点（无放回）。
    注意 Open3D 接口 random_down_sample 的参数是"采样比例 ratio∈(0,1)"，不是点数，
    这里按 n/N 换算成比例。
    优点：极快；缺点：不保证空间均匀，可能出现空洞、丢失小目标。"""
    t0 = time.time()
    n = min(n, len(pcd.points))
    ratio = float(n) / float(len(pcd.points))
    ratio = min(max(ratio, 1e-9), 0.999999)
    out = pcd.random_down_sample(ratio)
    return out, time.time() - t0


# ------------------------------------------------ 4. Keypoint (feature preserving)
def keypoint_downsample(pcd, salient_radius, non_max_radius, gamma_21=0.975, gamma_32=0.975):
    """关键点下采样（特征保持型）：ISS(Intrinsic Shape Signature) 关键点。

    原理：对每个点用邻域协方差矩阵的特征值 λ1>=λ2>=λ3 判断其"显著性"
          λ3/λ2 与 λ2/λ1 同时满足阈值、且在邻域内是局部极大 -> 记为关键点。
    优点：保留角点/边缘/曲面突变等结构信息，点数大幅减少但形状可辨认；
    缺点：阈值敏感，平坦区域几乎不出点。
    """
    t0 = time.time()
    cloud = o3d.geometry.PointCloud(pcd)
    if not cloud.has_normals():
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=salient_radius * 2.0, max_nn=30))
    kp = o3d.geometry.keypoint.compute_iss_keypoints(
        cloud, salient_radius=salient_radius, non_max_radius=non_max_radius,
        gamma_21=gamma_21, gamma_32=gamma_32)
    return kp, time.time() - t0


# ------------------------------------------------ 5. FPS
def fps_downsample(pcd, n):
    """最远点采样：先取一个种子点，之后每步选"离已选点集最远"的点。
    优点：点分布最均匀、覆盖最好（点云网络常用）；缺点：O(N*K)，慢。"""
    t0 = time.time()
    n = min(n, len(pcd.points))
    out = pcd.farthest_point_down_sample(n)
    return out, time.time() - t0


# ----------------------------------------------------------------------------- 可视化
# 每朵点云一种颜色；弹窗里用文字图例说明「哪种颜色 = 哪种下采样方法」
# 注意：Open3D GUI 默认字体不含中文字形（会显示成 "?"），窗口内文字一律用 ASCII
_PALETTE = [
    ((0.85, 0.30, 0.30), "Red"),
    ((0.20, 0.45, 0.85), "Blue"),
    ((0.20, 0.70, 0.35), "Green"),
    ((0.90, 0.65, 0.10), "Orange"),
    ((0.55, 0.35, 0.75), "Purple"),
    ((0.25, 0.70, 0.75), "Cyan"),
]


def _palette(n):
    return np.array([_PALETTE[i % len(_PALETTE)][0] for i in range(n)], dtype=float)


def _color_name(i):
    return _PALETTE[i % len(_PALETTE)][1]


def _layout_tiles(clouds, cols):
    """按 cols 列把点云平移成网格（返回拷贝，不改动原数据）。"""
    spans = [c.get_axis_aligned_bounding_box().get_extent() for c in clouds]
    gx = max(s[0] for s in spans) * 1.35
    gy = max(s[1] for s in spans) * 1.35
    views = []
    for i, c in enumerate(clouds):
        view = o3d.geometry.PointCloud(c)
        row, col = divmod(i, cols)
        view.translate((gx * col, -gy * row, 0.0))
        views.append(view)
    return views


_gui_ready = False


def _gui_app():
    """初始化（只做一次）Open3D GUI Application。"""
    global _gui_ready
    from open3d.visualization import gui

    app = gui.Application.instance
    if not _gui_ready:
        app.initialize()
        _gui_ready = True
    return app


def _create_compare_window(clouds, names, window_name, point_size, cols):
    """建立对比窗口：左侧文字图例（颜色 → 方法），右侧按网格并排显示各方法结果，
    并在每朵点云上方再加一条 3D 文字标注。"""
    from open3d.visualization import gui
    from open3d.visualization.rendering import Open3DScene, MaterialRecord

    app = _gui_app()
    colors = _palette(len(clouds))
    views = _layout_tiles(clouds, cols)

    win = app.create_window(window_name, 1680, 960)

    legend = gui.Vert(6, gui.Margins(16, 18, 16, 18))
    legend.background_color = gui.Color(0.12, 0.12, 0.15)
    legend.add_child(gui.Label("Color -> Downsample Method"))
    legend.add_child(gui.Label("-" * 30))
    for i, (c, n) in enumerate(zip(clouds, names)):
        item = gui.Label(f"[*] {_color_name(i):<7s}: {n:<12s} ({len(c.points)} pts)")
        item.text_color = gui.Color(*colors[i])
        legend.add_child(item)
    legend.add_child(gui.Label(""))
    hint = gui.Label("Left drag : rotate\nWheel     : zoom\n"
                     "Middle / Shift+drag : pan")
    hint.text_color = gui.Color(0.65, 0.65, 0.68)
    legend.add_child(hint)

    scene_widget = gui.SceneWidget()
    scene_widget.scene = Open3DScene(win.renderer)
    scene_widget.scene.set_background(np.array([0.08, 0.08, 0.10, 1.0]))

    for i, view in enumerate(views):
        view.paint_uniform_color(colors[i])
        mat = MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = float(point_size)
        scene_widget.scene.add_geometry(f"cloud_{i}", view, mat)

    def _on_layout(ctx):
        r = win.content_rect
        lw = int(legend.calc_preferred_size(ctx, gui.Widget.Constraints()).width) + 12
        legend.frame = gui.Rect(r.x, r.y, lw, r.height)
        scene_widget.frame = gui.Rect(r.x + lw, r.y, r.width - lw, r.height)

    win.set_on_layout(_on_layout)
    win.add_child(legend)
    win.add_child(scene_widget)

    bounds = scene_widget.scene.bounding_box
    scene_widget.setup_camera(60.0, bounds, bounds.get_center())

    for i, (view, name) in enumerate(zip(views, names)):
        bbox = view.get_axis_aligned_bounding_box()
        lo, hi = bbox.get_min_bound(), bbox.get_max_bound()
        label_pos = ((lo[0] + hi[0]) * 0.5, hi[1], (lo[2] + hi[2]) * 0.5)
        scene_widget.add_3d_label(label_pos, f"{_color_name(i)} {name} ({len(view.points)} pts)")

    return win, scene_widget


def compare_window(clouds, names, window_name="Downsample Compare", point_size=2.5, cols=3):
    """弹窗并排对比：窗口内用文字标注「哪种颜色对应哪种下采样方法」。"""
    if not clouds:
        return
    try:
        _create_compare_window(clouds, names, window_name, point_size, cols)
    except Exception as exc:                       # 无显示设备 / GUI 不可用
        print(f"[WARN] 无法创建可视化窗口（{exc}），请加 --no-show 仅看对比图。")
        return

    print(f"[INFO] 窗口已打开（{window_name}），颜色与方法的对应关系：")
    for i, (c, n) in enumerate(zip(clouds, names)):
        print(f"        {_color_name(i):<7s}-> {n}  ({len(c.points)} pts)")
    print("[INFO] 左键旋转 / 滚轮缩放 / 中键(或Shift+左键)平移 / 关闭窗口退出")
    _gui_app().run()


def save_compare_fig(clouds, names, path="downsample_compare.png", cols=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(clouds)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(6.0 * cols, 4.8 * rows), dpi=130)
    colors = _palette(n)
    for i, (c, name) in enumerate(zip(clouds, names)):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        pts = np.asarray(c.points)
        step = max(1, len(pts) // 60000)
        ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
                   s=1.5, c=[colors[i]], linewidths=0, alpha=0.85)
        ax.set_title(f"{name}\npoints = {len(pts)}", fontsize=10)
        ax.set_box_aspect((np.ptp(pts[:, 0]) + 1e-6,
                           np.ptp(pts[:, 1]) + 1e-6,
                           np.ptp(pts[:, 2]) + 1e-6))
        ax.view_init(elev=22, azim=-62)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
    fig.suptitle("3D Point Cloud Downsampling: Voxel / Octree / Random / ISS Keypoint / FPS", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] 对比图已保存：{os.path.abspath(path)}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Open3D 点云下采样：Voxel / Octree / Random / ISS / FPS")
    ap.add_argument("-i", "--input", default=DEFAULT_PCD,
                    help="输入 pcd 文件名（默认在脚本同级 data/ 目录下查找）")
    ap.add_argument("-o", "--outdir", default="output_downsample", help="输出目录")
    ap.add_argument("--voxel", type=float, default=0.01, help="体素边长(米)，八叉树用同一分辨率")
    ap.add_argument("--target", type=int, default=None,
                    help="随机/FPS 的目标点数，默认取体素结果的点数以便公平对比")
    ap.add_argument("--iss-salient", type=float, default=None, help="ISS 显著半径，默认 1.5*体素")
    ap.add_argument("--iss-nms", type=float, default=None, help="ISS 非极大抑制半径，默认 1.0*体素")
    ap.add_argument("--no-show", action="store_true", help="不弹 Open3D 窗口")
    ap.add_argument("--no-fig", action="store_true", help="不保存 matplotlib 对比图")
    ap.add_argument("--fig", default="downsample_compare.png", help="对比图文件名")
    ap.add_argument("--point-size", type=float, default=2.5, help="窗口点大小")
    args = ap.parse_args()

    path = resolve_pcd(args.input)
    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]

    print("=" * 82)
    src = load_cloud(path)
    print(f"输入点云 : {os.path.abspath(path)}")
    print(f"原始点数 : {len(src.points)}   包围盒 : "
          f"{src.get_axis_aligned_bounding_box().get_extent().round(3)}   体素边长 : {args.voxel} m")

    results = []                                   # (name, cloud, secs, note)

    # 1 Voxel Grid
    vg, t = voxel_downsample(src, args.voxel)
    results.append(("VoxelGrid", vg, t, f"voxel={args.voxel}"))

    # 2 Octree
    oc, t = octree_downsample(src, resolution=args.voxel, mode="center")
    oc_c, t2 = octree_downsample(src, resolution=args.voxel, mode="centroid")
    results.append(("Octree", oc, t, f"res={args.voxel}, 叶子体素中心"))

    # 目标点数：默认对齐体素结果
    target = args.target or len(vg.points)

    # 3 Random
    rd, t = random_downsample(src, target)
    results.append(("Random", rd, t, f"n={target}"))

    # 4 ISS Keypoint
    r_sal = args.iss_salient or args.voxel * 1.5
    r_nms = args.iss_nms or args.voxel * 1.0
    kp, t = keypoint_downsample(src, r_sal, r_nms)
    results.append(("ISS Keypoint", kp, t, f"salient={r_sal:.4f}, nms={r_nms:.4f}"))

    # 5 FPS
    fp, t = fps_downsample(src, target)
    results.append(("FPS", fp, t, f"n={target}"))

    # ----------------------------------------------------------------- 报表
    print("-" * 82)
    print(f"{'Method':<16}{'kept':>9}{'ratio':>9}{'time(s)':>10}   params")
    for name, cloud, secs, note in results:
        keep = len(cloud.points)
        print(f"{name:<16}{keep:>9d}{len(src.points) / max(keep, 1):>8.2f}x{secs:>10.2f}   {note}")
    print(f"{'(Octree质点)':<16}{len(oc_c.points):>9d}   <- mode=centroid，与 VoxelGrid 对照用")
    print("-" * 82)
    print("要点：VoxelGrid 与 Octree 点数/形状接近但分组不同；Random 有空洞；"
          "ISS 只保留结构特征点；FPS 分布最均匀但最慢。")

    # ----------------------------------------------------------------- 落盘
    files = {
        f"{stem}_orig.pcd": src,
        f"{stem}_voxel.pcd": vg,
        f"{stem}_octree.pcd": oc,
        f"{stem}_random.pcd": rd,
        f"{stem}_keypoint_iss.pcd": kp,
        f"{stem}_fps.pcd": fp,
    }
    for fn, cloud in files.items():
        o3d.io.write_point_cloud(os.path.join(args.outdir, fn), cloud, write_ascii=False)
    print(f"[INFO] 点云已写入 {os.path.abspath(args.outdir)}/")

    clouds = [src, vg, oc, rd, kp, fp]
    names = ["Original", "VoxelGrid", "Octree", "Random", "ISS Keypoint", "FPS"]
    if not args.no_fig:
        save_compare_fig(clouds, names, args.fig)
    if not args.no_show:
        compare_window(clouds, names, point_size=args.point_size)
    print("=" * 82)


if __name__ == "__main__":
    main()
#（注：内容由AI生成）
