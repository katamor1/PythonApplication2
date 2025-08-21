import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math

# ===================================================================
# ⚙️ 初期設定値 (GUIから変更可能)
# ===================================================================
SETTINGS = {
    # 幾何公差・オフセット
    "A_AX_OFFSET_Y": 0.0,
    "A_AX_OFFSET_Z": -5.0,
    "C_AX_ERROR_X": 0.5,
    "C_AX_ERROR_Y": 0.2,
    # 基準角度定数 (R_0, A_0, C_0)
    "ZERO_ANGLE_R": 0.0,
    "ZERO_ANGLE_A": 23.4,
    "ZERO_ANGLE_C": 12.3,
    # ワーク（四角柱）の定義
    "WORK_OFS_X": 2.0,
    "WORK_OFS_Y": 1.0,
    "WORK_OFS_Z": 0.5,
    "WORK_WIDTH": 1.0,
    "WORK_DEPTH": 1.0,
    "WORK_HEIGHT": 1.0,
    # テーブルの定義
    "TABLE_RADIUS": 5.0,
}
# ===================================================================

# --- 日本語フォントの設定 ---
plt.rcParams['font.family'] = 'MS Gothic'

# ---- グローバル変数 ----
ax = None
prism_poly3d = None
table_poly3d = None
tool_tip_plot = None
tool_cyl_plot = None
slider_va = None
slider_vc = None
text_boxes = {}

# ---- ボールエンドミル工具の形状を定義する関数 ----
def create_tool_geometry(contact_point, radius=0.5, length=8.0):
    phi = np.linspace(0, np.pi / 2, 15); theta = np.linspace(0, 2 * np.pi, 30)
    phi, theta = np.meshgrid(phi, theta)
    x_tip = radius * np.sin(phi) * np.cos(theta) + contact_point[0]
    y_tip = radius * np.sin(phi) * np.sin(theta) + contact_point[1]
    z_tip = -radius * np.cos(phi) + contact_point[2]
    z_cyl = np.linspace(0, length, 15); theta_cyl = np.linspace(0, 2 * np.pi, 30)
    theta_cyl, z_cyl = np.meshgrid(theta_cyl, z_cyl)
    x_cyl = radius * np.cos(theta_cyl) + contact_point[0]
    y_cyl = radius * np.sin(theta_cyl) + contact_point[1]
    z_cyl += contact_point[2]
    return (x_tip, y_tip, z_tip), (x_cyl, y_cyl, z_cyl)

# ---- 回転行列を定義する関数 ----
def rotation_matrix_x(theta_deg):
    theta_rad = np.deg2rad(theta_deg)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rotation_matrix_z(theta_deg):
    theta_rad = np.deg2rad(theta_deg)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# ---- 形状定義の関数 ----
def define_prism_local(size):
    w, d, h = size
    x = np.array([-w/2, w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2])
    y = np.array([-d/2, -d/2, d/2, d/2, -d/2, -d/2, d/2, d/2])
    z = np.array([-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2])
    return np.vstack([x, y, z]).T

def get_circle_surface(center, radius, num_points=100):
    cx, cy, cz = center
    theta = np.linspace(0, 2 * np.pi, num_points)
    verts = np.array([cx + radius * np.cos(theta), cy + radius * np.sin(theta), np.full(num_points, cz)]).T
    faces = [[verts[i], verts[i-1], [cx, cy, cz]] for i in range(num_points)]
    return verts, faces

# ---- メインの更新・描画関数 ----
def update(val):
    global prism_poly3d, table_poly3d, text_boxes, tool_tip_plot, tool_cyl_plot
    
    # --- 1. 逆運動学計算 ---
    # スライダーから指令角度 Av, Cv を取得
    Av = slider_va.val
    Cv = slider_vc.val
    
    # 定数 R_0, A_0, C_0 を取得
    R_0 = SETTINGS["ZERO_ANGLE_R"]
    A_0 = SETTINGS["ZERO_ANGLE_A"]
    C_0 = SETTINGS["ZERO_ANGLE_C"]

    # Z軸方向単位ベクトル Vz
    Vz = np.array([0, 0, 1])

    # 指令角度からターゲットベクトルを計算: V_target = Rz(Cv)Rx(Av)Vz
    R_Av = rotation_matrix_x(Av)
    R_Cv = rotation_matrix_z(Cv)
    V_target = R_Cv @ R_Av @ Vz
    
    # 方程式の左辺から R_machine を分離するための変換行列を計算
    # (Rz(-R_0)Rx(-A_0)Rz(-C_0))^-1 = Rz(C_0)Rx(A_0)Rz(R_0)
    R_R0 = rotation_matrix_z(R_0)
    R_A0 = rotation_matrix_x(A_0)
    R_C0 = rotation_matrix_z(C_0)
    R_mount_for_IK = R_C0 @ R_A0 @ R_R0
    
    # 機械座標系での最終的な工具ベクトルを計算
    V_final = R_mount_for_IK @ V_target
    
    # V_final = [x, y, z] から A と C を解析的に解く
    x, y, z = V_final
    
    # A = arccos(z)
    A_rad = np.arccos(np.clip(z, -1.0, 1.0)) # 値を-1から1の範囲にクリップしてエラーを防ぐ
    
    # C = atan2(x, -y)
    C_rad = np.arctan2(x, -y)
        
    A_deg, C_deg = np.rad2deg([A_rad, C_rad])
    
    # 計算結果をGUIに表示
    text_boxes["Machine A"].set_val(f"{A_deg:.2f}")
    text_boxes["Machine C"].set_val(f"{C_deg:.2f}")
    
    # 法線ベクトルの表示 (これは仮想的なもの)
    text_boxes["Normal X"].set_val(f"{V_target[0]:.4f}")
    text_boxes["Normal Y"].set_val(f"{V_target[1]:.4f}")
    text_boxes["Normal Z"].set_val(f"{V_target[2]:.4f}")

    # --- 2. 順運動学 & 描画更新 ---
    R_a, R_c = rotation_matrix_x(A_deg), rotation_matrix_z(C_deg)
    
    # rebuild_plotと計算方法を統一
    R_mount_Fwd = R_C0.T @ R_A0.T @ R_R0.T
    prism_local_vertices = define_prism_local((SETTINGS["WORK_WIDTH"], SETTINGS["WORK_DEPTH"], SETTINGS["WORK_HEIGHT"]))
    prism_mounted_vertices = (R_mount_Fwd @ prism_local_vertices.T).T
    prism_initial_center = np.array([
        SETTINGS["C_AX_ERROR_X"] + SETTINGS["WORK_OFS_X"],
        SETTINGS["C_AX_ERROR_Y"] + SETTINGS["WORK_OFS_Y"],
        SETTINGS["WORK_OFS_Z"]
    ])
    original_prism_vertices = prism_mounted_vertices + prism_initial_center
    table_initial_center = (SETTINGS["C_AX_ERROR_X"], SETTINGS["C_AX_ERROR_Y"], 0)
    original_table_verts, _ = get_circle_surface(table_initial_center, SETTINGS["TABLE_RADIUS"])
    offset_vector = np.array([0, SETTINGS["A_AX_OFFSET_Y"], SETTINGS["A_AX_OFFSET_Z"]])
    
    # プリズムの変換 
    prism_c_rotated = (R_c @ original_prism_vertices.T).T
    prism_offset = prism_c_rotated + offset_vector
    prism_final = (R_a @ prism_offset.T).T

    # テーブルの変換 
    table_c_rotated_verts = (R_c @ original_table_verts.T).T
    table_offset_verts = table_c_rotated_verts + offset_vector
    table_final_verts = (R_a @ table_offset_verts.T).T
    
    # 頂点データ更新
    faces_indices = [[0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 2, 3], [4, 5, 6, 7]]
    prism_poly3d.set_verts([prism_final[i] for i in faces_indices])
    
    table_center_final = R_a @ ( (R_c @ table_initial_center) + offset_vector)
    final_faces = [[v1, v2, table_center_final] for v1, v2 in zip(table_final_verts, np.roll(table_final_verts, -1, axis=0))]
    table_poly3d.set_verts(final_faces)

    # --- 3. 工具の位置を計算して再描画 ---
    top_center_local = np.array([0, 0, SETTINGS["WORK_HEIGHT"] / 2])
    top_center_mounted = (R_mount_Fwd @ top_center_local.T).T
    original_top_center = top_center_mounted + prism_initial_center
    
    top_center_c_rotated = (R_c @ original_top_center.T).T
    top_center_offset = top_center_c_rotated + offset_vector
    top_center_final = (R_a @ top_center_offset.T).T
    
    if tool_tip_plot: tool_tip_plot.remove()
    if tool_cyl_plot: tool_cyl_plot.remove()
        
    tip_geom, cyl_geom = create_tool_geometry(top_center_final)
    tool_tip_plot = ax.plot_surface(tip_geom[0], tip_geom[1], tip_geom[2], color='gray', alpha=0.9, zorder=10)
    tool_cyl_plot = ax.plot_surface(cyl_geom[0], cyl_geom[1], cyl_geom[2], color='dimgray', alpha=0.9, zorder=10)
    
    fig.canvas.draw_idle()

# ---- GUIの再構築と設定適用 ----
def apply_settings(event):
    global SETTINGS, text_boxes
    for key, textbox in text_boxes.items():
        if key in SETTINGS:
            try:
                SETTINGS[key] = float(textbox.text)
            except ValueError:
                print(f"Warning: Invalid value for {key}. Using previous value.")
    rebuild_plot()

def rebuild_plot():
    global ax, prism_poly3d, table_poly3d, tool_tip_plot, tool_cyl_plot
    ax.clear()

    R_r_zero = rotation_matrix_z(SETTINGS["ZERO_ANGLE_R"])
    R_a_zero = rotation_matrix_x(SETTINGS["ZERO_ANGLE_A"])
    R_c_zero = rotation_matrix_z(SETTINGS["ZERO_ANGLE_C"])
    R_mount = R_c_zero.T @ R_a_zero.T @ R_r_zero.T # 順運動学での取り付け姿勢
    
    prism_local_vertices = define_prism_local((SETTINGS["WORK_WIDTH"], SETTINGS["WORK_DEPTH"], SETTINGS["WORK_HEIGHT"]))
    prism_mounted_vertices = (R_mount @ prism_local_vertices.T).T
    prism_initial_center = np.array([
        SETTINGS["C_AX_ERROR_X"] + SETTINGS["WORK_OFS_X"],
        SETTINGS["C_AX_ERROR_Y"] + SETTINGS["WORK_OFS_Y"],
        SETTINGS["WORK_OFS_Z"]
    ])
    original_prism_vertices = prism_mounted_vertices + prism_initial_center
    table_initial_center = (SETTINGS["C_AX_ERROR_X"], SETTINGS["C_AX_ERROR_Y"], 0)
    original_table_verts, original_table_faces = get_circle_surface(table_initial_center, SETTINGS["TABLE_RADIUS"])

    faces_indices = [[0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 2, 3], [4, 5, 6, 7]]
    face_colors = ['cyan'] * 5 + ['yellow']
    prism_poly3d = Poly3DCollection([original_prism_vertices[i] for i in faces_indices],
                                      facecolors=face_colors, linewidths=1, edgecolors='k', alpha=0.9)
    ax.add_collection3d(prism_poly3d)
    table_poly3d = Poly3DCollection(original_table_faces, facecolors='gray', linewidths=0, alpha=0.3)
    ax.add_collection3d(table_poly3d)
    
    tool_tip_plot, tool_cyl_plot = None, None

    ax.set_title('5-Axis Trunnion Simulation')
    ax.set_xlabel('X-axis'), ax.set_ylabel('Y-axis'), ax.set_zlabel('Z-axis')
    limit = max(SETTINGS["TABLE_RADIUS"] * 1.2, 5)
    ax.set_xlim([-limit, limit]), ax.set_ylim([-limit, limit]), ax.set_zlim([-limit, limit])
    ax.plot([-limit, limit], [0, 0], [0, 0], color='red', linestyle='--', linewidth=1, label='A-axis (X)')
    ax.plot([0, 0], [0, 0], [-limit, limit], color='blue', linestyle='--', linewidth=1, label='C-axis (Z)')
    ax.legend()
    ax.view_init(elev=25., azim=45)
    ax.set_aspect('equal', adjustable='box')
    update(0)

# ---- メイン処理 ----
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
fig.subplots_adjust(bottom=0.45, top=0.95, left=0.05, right=0.95)

# --- GUIコントロールの配置 ---
ax_slider_va = fig.add_axes([0.1, 0.35, 0.8, 0.03])
slider_va = Slider(ax=ax_slider_va, label='Virtual A (Av) [deg]', valmin=-180, valmax=180, valinit=0)
ax_slider_vc = fig.add_axes([0.1, 0.30, 0.8, 0.03])
slider_vc = Slider(ax=ax_slider_vc, label='Virtual C (Cv) [deg]', valmin=-180, valmax=180, valinit=0)

info_labels = ["Machine A", "Machine C", "Normal X", "Normal Y", "Normal Z"]
for i, label in enumerate(info_labels):
    ax_box = fig.add_axes([0.1 + i*0.18, 0.22, 0.12, 0.03])
    text_boxes[label] = TextBox(ax_box, label, initial="0.0")

setting_keys = list(SETTINGS.keys())
for i, key in enumerate(setting_keys):
    ax_box = fig.add_axes([0.1 + (i % 4) * 0.22, 0.14 - (i // 4) * 0.04, 0.08, 0.03])
    text_boxes[key] = TextBox(ax_box, key, initial=str(SETTINGS[key]))

ax_button = fig.add_axes([0.85, 0.01, 0.1, 0.03])
apply_button = Button(ax_button, 'Apply Settings')
apply_button.on_clicked(apply_settings)

slider_va.on_changed(update)
slider_vc.on_changed(update)

rebuild_plot()
plt.show()