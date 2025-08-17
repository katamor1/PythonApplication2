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
    # 取り付け姿勢の基準角度
    "ZERO_ANGLE_R": 90.0,
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

# ---- グローバル変数 (プロットオブジェクト) ----
ax = None
prism_poly3d = None
table_poly3d = None
slider_va = None
slider_vc = None
text_boxes = {} # GUIのTextBoxオブジェクトを保持

# ---- 回転行列を定義する関数 ----
def rotation_matrix_x(theta_deg):
    theta_rad = np.deg2rad(theta_deg)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rotation_matrix_y(theta_deg):
    theta_rad = np.deg2rad(theta_deg)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

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
    global prism_poly3d, table_poly3d, text_boxes
    
    # --- 1. 逆運動学計算 ---
    R_vA = rotation_matrix_x(slider_va.val)
    R_vC = rotation_matrix_z(slider_vc.val)
    R_target_final = R_vA @ R_vC

    initial_normal = np.array([0, 0, 1])
    current_normal = R_target_final @ initial_normal
    text_boxes["Normal X"].set_val(f"{current_normal[0]:.4f}")
    text_boxes["Normal Y"].set_val(f"{current_normal[1]:.4f}")
    text_boxes["Normal Z"].set_val(f"{current_normal[2]:.4f}")
    
    R_r_zero = rotation_matrix_z(SETTINGS["ZERO_ANGLE_R"])
    R_a_zero = rotation_matrix_x(SETTINGS["ZERO_ANGLE_A"])
    R_c_zero = rotation_matrix_z(SETTINGS["ZERO_ANGLE_C"])
    R_mount_inv = R_r_zero @ R_a_zero @ R_c_zero
    R_machine_target = R_target_final @ R_mount_inv
    
    sin_A = -R_machine_target[1, 2]
    cos_A = math.sqrt(R_machine_target[1, 0]**2 + R_machine_target[1, 1]**2)
    A_rad = math.atan2(sin_A, cos_A)
    
    if np.isclose(cos_A, 0):
        C_rad = 0
        B_rad = math.atan2(-R_machine_target[2, 0], R_machine_target[0, 0])
    else:
        C_rad = math.atan2(R_machine_target[1, 0], R_machine_target[1, 1])
        B_rad = math.atan2(R_machine_target[0, 2], R_machine_target[2, 2])
        
    A_deg, B_deg, C_deg = np.rad2deg([A_rad, B_rad, C_rad])
    text_boxes["Machine A"].set_val(f"{A_deg:.2f}")
    text_boxes["Machine B"].set_val(f"{B_deg:.2f}")
    text_boxes["Machine C"].set_val(f"{C_deg:.2f}")

    # --- 2. 順運動学 & 描画更新 ---
    R_a, R_b, R_c = rotation_matrix_x(A_deg), rotation_matrix_y(B_deg), rotation_matrix_z(C_deg)
    
    # ワークとテーブルの初期形状を再計算
    R_mount = R_c_zero.T @ R_a_zero.T
    prism_local_vertices = define_prism_local((SETTINGS["WORK_WIDTH"], SETTINGS["WORK_DEPTH"], SETTINGS["WORK_HEIGHT"]))
    prism_mounted_vertices = (R_mount @ prism_local_vertices.T).T
    prism_initial_center = np.array([
        SETTINGS["C_AX_ERROR_X"] + SETTINGS["WORK_OFS_X"],
        SETTINGS["C_AX_ERROR_Y"] + SETTINGS["WORK_OFS_Y"],
        SETTINGS["WORK_OFS_Z"]
    ])
    original_prism_vertices = prism_mounted_vertices + prism_initial_center
    table_initial_center = (SETTINGS["C_AX_ERROR_X"], SETTINGS["C_AX_ERROR_Y"], 0)
    original_table_verts, _ = get_circle_surface(table_initial_center, SETTINGS["TABLE_RADIUS"])

    # 座標変換を安全なステップ・バイ・ステップ方式で適用
    offset_vector = np.array([0, SETTINGS["A_AX_OFFSET_Y"], SETTINGS["A_AX_OFFSET_Z"]])
    
    # プリズムの変換
    prism_c_rotated = (R_c @ original_prism_vertices.T).T
    prism_offset = prism_c_rotated + offset_vector
    prism_a_rotated = (R_a @ prism_offset.T).T
    prism_final = (R_b @ prism_a_rotated.T).T

    # テーブルの変換
    table_c_rotated_verts = (R_c @ original_table_verts.T).T
    table_offset_verts = table_c_rotated_verts + offset_vector
    table_a_rotated_verts = (R_a @ table_offset_verts.T).T
    table_final_verts = (R_b @ table_a_rotated_verts.T).T
    
    # 頂点データ更新
    faces_indices = [[0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 2, 3], [4, 5, 6, 7]]
    prism_poly3d.set_verts([prism_final[i] for i in faces_indices])
    
    table_center_final = (R_b @ (R_a @ ( (R_c @ table_initial_center) + offset_vector)))
    final_faces = [[v1, v2, table_center_final] for v1, v2 in zip(table_final_verts, np.roll(table_final_verts, -1, axis=0))]
    table_poly3d.set_verts(final_faces)
    
    fig.canvas.draw_idle()

# ---- GUIの再構築と設定適用 ----
def apply_settings(event):
    """設定テキストボックスの値を読み込み、プロットを再構築する"""
    global SETTINGS, text_boxes
    
    # GUIから値を取得してSETTINGSを更新
    for key, textbox in text_boxes.items():
        if key in SETTINGS:
            try:
                SETTINGS[key] = float(textbox.text)
            except ValueError:
                print(f"Warning: Invalid value for {key}. Using previous value.")
    
    rebuild_plot()

def rebuild_plot():
    """現在の設定値に基づいてプロット全体をクリアし、再描画する"""
    global ax, prism_poly3d, table_poly3d
    
    ax.clear()

    # オブジェクトの初期状態を計算
    R_r_zero = rotation_matrix_z(SETTINGS["ZERO_ANGLE_R"])
    R_a_zero = rotation_matrix_x(SETTINGS["ZERO_ANGLE_A"])
    R_c_zero = rotation_matrix_z(SETTINGS["ZERO_ANGLE_C"])
    R_mount = R_c_zero.T @ R_a_zero.T @ R_r_zero.T
    
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

    # 3Dオブジェクトを再作成してプロットに追加
    faces_indices = [[0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 2, 3], [4, 5, 6, 7]]
    face_colors = ['cyan'] * 5 + ['yellow']
    prism_poly3d = Poly3DCollection([original_prism_vertices[i] for i in faces_indices],
                                    facecolors=face_colors, linewidths=1, edgecolors='k', alpha=0.9)
    ax.add_collection3d(prism_poly3d)

    table_poly3d = Poly3DCollection(original_table_faces, facecolors='gray', linewidths=0, alpha=0.3)
    ax.add_collection3d(table_poly3d)

    # プロットの見た目を設定
    ax.set_title('6-Axis Trunnion Simulation')
    ax.set_xlabel('X-axis'), ax.set_ylabel('Y-axis'), ax.set_zlabel('Z-axis')
    limit = max(SETTINGS["TABLE_RADIUS"] * 1.2, 5)
    ax.set_xlim([-limit, limit]), ax.set_ylim([-limit, limit]), ax.set_zlim([-limit, limit])
    ax.plot([-limit, limit], [0, 0], [0, 0], color='red', linestyle='--', linewidth=1, label='A-axis (X)')
    ax.plot([0, 0], [-limit, limit], [0, 0], color='green', linestyle='--', linewidth=1, label='B-axis (Y)')
    ax.plot([0, 0], [0, 0], [-limit, limit], color='blue', linestyle='--', linewidth=1, label='C-axis (Z)')
    ax.legend()
    ax.view_init(elev=25., azim=45)
    ax.set_aspect('equal', adjustable='box')

    # 初期描画
    update(0)

# ---- メイン処理 ----
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
fig.subplots_adjust(bottom=0.45, top=0.95, left=0.05, right=0.95)

# --- GUIコントロールの配置 ---
# 仮想軸スライダー
ax_slider_va = fig.add_axes([0.1, 0.35, 0.8, 0.03])
slider_va = Slider(ax=ax_slider_va, label='Virtual A [deg]', valmin=-180, valmax=180, valinit=0)
ax_slider_vc = fig.add_axes([0.1, 0.30, 0.8, 0.03])
slider_vc = Slider(ax=ax_slider_vc, label='Virtual C [deg]', valmin=-180, valmax=180, valinit=0)

# 機械軸と法線ベクトルの表示
info_labels = ["Machine A", "Machine B", "Machine C", "Normal X", "Normal Y", "Normal Z"]
for i, label in enumerate(info_labels):
    ax_box = fig.add_axes([0.1 + (i % 3) * 0.2, 0.22 - (i // 3) * 0.04, 0.12, 0.03])
    text_boxes[label] = TextBox(ax_box, label, initial="0.0")

# 設定値の入力ボックス
setting_keys = list(SETTINGS.keys())
for i, key in enumerate(setting_keys):
    ax_box = fig.add_axes([0.1 + (i % 4) * 0.22, 0.14 - (i // 4) * 0.04, 0.08, 0.03])
    text_boxes[key] = TextBox(ax_box, key, initial=str(SETTINGS[key]))

# 設定適用ボタン
ax_button = fig.add_axes([0.85, 0.01, 0.1, 0.03])
apply_button = Button(ax_button, 'Apply Settings')
apply_button.on_clicked(apply_settings)

# スライダーのイベントに関数を接続
slider_va.on_changed(update)
slider_vc.on_changed(update)

# 初回描画
rebuild_plot()
plt.show()