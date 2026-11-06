#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
open3d_filter.py —— 3D 点云滤波（离群点去除）：SOR 统计滤波 + ROR 半径滤波（Open3D 单文件版）

功能
  1. 读取本地 PCD（默认用 data/table_scene_mug_stereo_textured.pcd）
  2. SOR  Statistical Outlier Removal  统计离群点滤波（对应 Open3D remove_statistical_outlier）
  3. ROR  Radius Outlier Removal       半径离群点滤波（对应 Open3D remove_radius_outlier）
  4. 打印过滤前后点数、剔除比例、耗时
  5. 结果落盘 ppf：*_original.pcd / *_sor.pcd / *_ror.pcd
  6. 弹出对比窗口：原始 / SOR / ROR / ROR+SOR 并排显示，
     左侧文字图例标明「颜色 → 滤波方法」（--no-show 可关闭）
     同时保存 matplotlib 2x2 对比图（headless/服务器环境也能出图）

用法
  python open3d_filter.py                                  # data/ 下默认点云 + 默认参数 + 开窗口
  python open3d_filter.py -i bunny.pcd --voxel-hint 0.005  # 换 data/bunny.pcd
  python open3d_filter.py --no-show --fig filter_compare.png
  python open3d_filter.py --k 30 --std 2.0 --r-points 10 --r-scale 4   # 调参

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
    # 结构化点云（WIDTH=640）常带 NaN，先清掉，否则邻域搜索会报错
    pts = np.asarray(pcd.points)
    bad = ~np.isfinite(pts).all(axis=1)
    if bad.any():
        print(f"[INFO] 发现 {int(bad.sum())} 个无效点(NaN/Inf)，已剔除；有效点 {len(pts) - int(bad.sum())}")
        pcd = pcd.select_by_index(np.where(~bad)[0])
    return pcd


def mean_nn_distance(pcd: o3d.geometry.PointCloud) -> float:
    """平均最近邻距离，用来估算点云平均点间距（半径滤波的合理半径来源）。"""
    return float(np.mean(pcd.compute_nearest_neighbor_distance()))


# ----------------------------------------------------------------------------- 滤波
def sor_filter(pcd: o3d.geometry.PointCloud, nb_neighbors: int = 20, std_ratio: float = 2.0):
    """SOR 统计滤波。

    对每个点统计其 nb_neighbors 个最近邻的平均距离 d_i，全景点云得到 d ~ N(mu, sigma)，
    d_i > mu + std_ratio * sigma 的点判为离群点剔除。
    """
    t0 = time.time()
    filtered, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                   std_ratio=std_ratio)
    return filtered, ind, time.time() - t0


def ror_filter(pcd: o3d.geometry.PointCloud, nb_points: int = 8, radius: float = 0.01):
    """ROR 半径滤波。

    以每个点为中心、radius 为半径的球内邻点数 < nb_points 的点判为离群点剔除。
    """
    t0 = time.time()
    filtered, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return filtered, ind, time.time() - t0


# ----------------------------------------------------------------------------- 可视化
# 每朵点云一种颜色；弹窗里用文字图例说明「哪种颜色 = 哪种处理方法」
# 注意：Open3D GUI 默认字体不含中文字形（会显示成 "?"），窗口内文字一律用 ASCII
_PALETTE = [
    ((0.85, 0.30, 0.30), "Red"),
    ((0.20, 0.45, 0.85), "Blue"),
    ((0.20, 0.70, 0.35), "Green"),
    ((0.90, 0.65, 0.10), "Orange"),
    ((0.55, 0.35, 0.75), "Purple"),
    ((0.25, 0.70, 0.75), "Cyan"),
]


def _palette(n: int):
    return np.array([_PALETTE[i % len(_PALETTE)][0] for i in range(n)], dtype=float)


def _color_name(i: int) -> str:
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
    """建立对比窗口：左侧文字图例（颜色 → 滤波方法），右侧并排显示各结果，
    并在每朵点云上方再加一条 3D 文字标注。"""
    from open3d.visualization import gui
    from open3d.visualization.rendering import Open3DScene, MaterialRecord

    app = _gui_app()
    colors = _palette(len(clouds))
    views = _layout_tiles(clouds, cols)

    win = app.create_window(window_name, 1680, 900)

    legend = gui.Vert(6, gui.Margins(16, 18, 16, 18))
    legend.background_color = gui.Color(0.12, 0.12, 0.15)
    legend.add_child(gui.Label("Color -> Filter Method"))
    legend.add_child(gui.Label("-" * 26))
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


def compare_window(clouds, names, window_name="Filter Compare", point_size=2.0, cols=None):
    """弹窗并排对比：窗口内用文字标注「哪种颜色对应哪种滤波方法」。"""
    if not clouds:
        return
    cols = cols or len(clouds)                       # 单行并排
    try:
        _create_compare_window(clouds, names, window_name, point_size, cols)
    except Exception as exc:                         # 无显示设备 / GUI 不可用
        print(f"[WARN] 无法创建可视化窗口（{exc}），请加 --no-show 仅出图。")
        return

    print(f"[INFO] 窗口已打开（{window_name}），颜色与方法的对应关系：")
    for i, (c, n) in enumerate(zip(clouds, names)):
        print(f"        {_color_name(i):<7s}-> {n}  ({len(c.points)} pts)")
    print("[INFO] 左键旋转 / 滚轮缩放 / 中键(或Shift+左键)平移 / 关闭窗口退出")
    _gui_app().run()


def save_compare_fig(clouds, names, path="filter_compare.png"):
    """matplotlib 六面/四面视角对比图（不需要 GL，服务器上也能出图）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(clouds)
    cols = 2 if n <= 4 else 3
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(6.0 * cols, 4.6 * rows), dpi=130)
    colors = _palette(n)
    for i, (c, name) in enumerate(zip(clouds, names)):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        pts = np.asarray(c.points)
        step = max(1, len(pts) // 60000)                     # 抽稀显示，避免卡顿
        ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
                   s=1.2, c=[colors[i]], linewidths=0, alpha=0.85)
        ax.set_title(f"{name}\npoints = {len(pts)}", fontsize=10)
        ax.set_box_aspect((np.ptp(pts[:, 0]) + 1e-6,
                           np.ptp(pts[:, 1]) + 1e-6,
                           np.ptp(pts[:, 2]) + 1e-6))
        ax.view_init(elev=22, azim=-62)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
    fig.suptitle("3D Point Cloud Filtering: Original / SOR / ROR", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] 对比图已保存：{os.path.abspath(path)}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Open3D 点云滤波：SOR 统计滤波 + ROR 半径滤波")
    ap.add_argument("-i", "--input", default=DEFAULT_PCD,
                    help="输入 pcd 文件名（默认在脚本同级 data/ 目录下查找）")
    ap.add_argument("-o", "--outdir", default="output_filter", help="结果输出目录")
    ap.add_argument("--k", type=int, default=30, help="SOR 邻域点数 nb_neighbors")
    ap.add_argument("--std", type=float, default=2.0, help="SOR 标准差倍数 std_ratio")
    ap.add_argument("--r-points", type=int, default=10, help="ROR 球内最少点数 nb_points")
    ap.add_argument("--r-scale", type=float, default=4.0,
                    help="ROR 半径 = r_scale * 平均最近邻距离")
    ap.add_argument("--radius", type=float, default=None, help="直接指定 ROR 半径(米)，优先级最高")
    ap.add_argument("--no-show", action="store_true", help="不弹 Open3D 窗口")
    ap.add_argument("--no-fig", action="store_true", help="不保存 matplotlib 对比图")
    ap.add_argument("--fig", default="filter_compare.png", help="对比图输出文件名")
    ap.add_argument("--point-size", type=float, default=2.0, help="窗口点大小")
    args = ap.parse_args()

    path = resolve_pcd(args.input)
    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]

    print("=" * 78)
    print(f"输入点云 : {os.path.abspath(path)}")
    src = load_cloud(path)
    nn = mean_nn_distance(src)
    print(f"原始点数 : {len(src.points)}    平均最近邻距离 : {nn:.4f} m    "
          f"包围盒 : {src.get_axis_aligned_bounding_box().get_extent().round(3)}")

    radius = args.radius if args.radius is not None else max(args.r_scale * nn, 1e-6)

    print("-" * 78)
    print(f"[1/2] SOR 统计滤波   nb_neighbors={args.k}  std_ratio={args.std}")
    sor, sor_idx, t_sor = sor_filter(src, args.k, args.std)
    print(f"      保留 {len(sor.points):>7d} 点，剔除 {len(src.points) - len(sor.points):>7d} 点 "
          f"({100 * (1 - len(sor.points) / len(src.points)):.2f}%)，耗时 {t_sor:.2f} s")

    print(f"[2/2] ROR 半径滤波   nb_points={args.r_points}  radius={radius:.5f} m")
    ror, ror_idx, t_ror = ror_filter(src, args.r_points, radius)
    print(f"      保留 {len(ror.points):>7d} 点，剔除 {len(src.points) - len(ror.points):>7d} 点 "
          f"({100 * (1 - len(ror.points) / len(src.points)):.2f}%)，耗时 {t_ror:.2f} s")

    # 两种滤波的交集（先 ROR 再 SOR 的组合式滤波，工程上常用）
    t0 = time.time()
    combo, _ = ror.remove_statistical_outlier(nb_neighbors=args.k, std_ratio=args.std)
    t_combo = time.time() - t0
    print(f"[+]  ROR -> SOR 组合  保留 {len(combo.points):>7d} 点，耗时 {t_combo:.2f} s")

    print("-" * 78)
    print(f"{'Method':<22}{'kept':>10}{'dropped':>10}{'drop%':>10}{'time(s)':>10}")
    for name, cloud, t in (("SOR", sor, t_sor), ("ROR", ror, t_ror), ("ROR + SOR", combo, t_combo)):
        keep, drop = len(cloud.points), len(src.points) - len(cloud.points)
        print(f"{name:<22}{keep:>10d}{drop:>10d}{100 * (1 - keep / len(src.points)):>9.2f}%{t:>10.2f}")

    # 落盘
    files = {
        f"{stem}_original.pcd": src,
        f"{stem}_sor.pcd": sor,
        f"{stem}_ror.pcd": ror,
        f"{stem}_ror_sor.pcd": combo,
    }
    for fn, cloud in files.items():
        o3d.io.write_point_cloud(os.path.join(args.outdir, fn), cloud, write_ascii=False)
    print(f"[INFO] 点云已写入 {os.path.abspath(args.outdir)}/  ({', '.join(files)})")

    clouds = [src, sor, ror, combo]
    names = ["Original", "SOR", "ROR", "ROR+SOR"]
    if not args.no_fig:
        save_compare_fig(clouds, names, args.fig)
    if not args.no_show:
        compare_window(clouds, names, point_size=args.point_size)
    print("=" * 78)


if __name__ == "__main__":
    main()
