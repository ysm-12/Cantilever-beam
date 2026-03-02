import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- 1. 页面基本配置 ---
st.set_page_config(page_title="Cantilever Beam Analysis", layout="wide")
st.title("Cantilever Beam Analysis Module（悬臂梁内力分析）")

# --- 2. 初始化 Session State (用于保存动态状态) ---
if 'unit_system' not in st.session_state:
    st.session_state.unit_system = 'SI'
if 'num_loads' not in st.session_state:
    st.session_state.num_loads = 2  # 【修改点1】：初始默认生成2个载荷输入框

# --- 3. 顶部控制栏 & 单位设置 ---
col_preset1, col_preset2, _ = st.columns([1, 1, 4])
with col_preset1:
    if st.button("Preset: SI (m, kN)"):
        st.session_state.unit_system = 'SI'
        st.rerun()
with col_preset2:
    if st.button("Preset: US (ft, kips)"):
        st.session_state.unit_system = 'US'
        st.rerun()

# 根据当前单位系统定义单位后缀
if st.session_state.unit_system == 'SI':
    u_len, u_force, u_moment, u_dist, u_E, u_I, u_def = "m", "kN", "kN·m", "kN/m", "GPa", "cm4", "mm"
else:
    u_len, u_force, u_moment, u_dist, u_E, u_I, u_def = "ft", "kips", "kips·ft", "kips/ft", "ksi", "in4", "in"

# --- 4. 界面：梁属性 (悬臂梁默认左端 x=0 处固定) ---
st.subheader("Beam Geometry (Fixed at Left End)")
col1, col2, col3 = st.columns(3)
# 【修改点2】：将 L 的 value 默认值修改为 36.0
L = col1.number_input(f"Length (L) [{u_len}]", min_value=0.1, value=36.0, step=1.0)
E_display = col2.number_input(f"Modulus (E) [{u_E}]", min_value=0.1, value=200.0, step=10.0)
I_display = col3.number_input(f"Inertia (I) [{u_I}]", min_value=0.1, value=5000.0, step=100.0)

# --- 5. 界面：载荷输入表 ---
st.subheader("Applied Loads")
col_add, col_reset, _ = st.columns([1, 1, 8])
with col_add:
    if st.button("➕ Add Load"):
        st.session_state.num_loads += 1
        st.rerun()
with col_reset:
    if st.button("🔄 Reset Loads"):
        st.session_state.num_loads = 2  # 重置时也保留2个
        st.rerun()

st.markdown(f"**Positive = Downward**")

loads_data = []
for i in range(st.session_state.num_loads):
    c1, c2, c3 = st.columns(3)

    # 【修改点3】：根据索引 i 分配默认的位置和大小
    if i == 0:
        default_pos = "13.5"
        default_mag = "125"
    elif i == 1:
        default_pos = "22.5"
        default_mag = "125"
    else:
        default_pos = ""
        default_mag = ""

    l_type = c1.selectbox("Type", ["Point Load (↓)", "Distributed Load (↓↓↓)"], key=f"type_{i}")
    pos = c2.text_input(f"Position x ({u_len})", value=default_pos, key=f"pos_{i}",
                        help="For dist load, can use 'start-end' like '0-5'")
    mag = c3.text_input(f"Magnitude", value=default_mag, key=f"mag_{i}")
    loads_data.append((l_type, pos, mag))

st.divider()

# --- 6. 核心计算与绘图逻辑 ---
n_pts = 1001
x = np.linspace(0, L, n_pts)
dx = L / (n_pts - 1)

point_forces = []
applied_forces = []
applied_dist = []

# 解析载荷
try:
    for l_type, s_pos, s_mag in loads_data:
        if not s_pos or not s_mag:
            continue

        mag = float(s_mag)

        if "Point" in l_type:
            pos = float(s_pos)
            if pos > L: pos = L
            applied_forces.append((pos, mag))
            point_forces.append((pos, mag))
        else:  # Distributed
            if "-" in s_pos:
                p1, p2 = map(float, s_pos.split("-"))
            else:
                start = float(s_pos)
                p1, p2 = start, L
            if p2 > L: p2 = L
            applied_dist.append((p1, p2, mag))
except ValueError:
    st.warning("⚠️ Please enter valid numbers for loads (e.g., '5' or '0-10').")
    st.stop()

# 计算固定端 (x=0) 的支座反力与反弯矩
force_sum = 0
moment_sum_about_A = 0

for pos, P in applied_forces:
    force_sum += P
    moment_sum_about_A += P * pos

for p1, p2, w in applied_dist:
    total_w = w * (p2 - p1)
    center = (p1 + p2) / 2
    force_sum += total_w
    moment_sum_about_A += total_w * center

R_A = force_sum
M_A = -moment_sum_about_A

# 显示反力结果
st.success(
    f"**Wall Reactions (at $x=0$):** Vertical Force $R_A = {R_A:.2f}$ {u_force} (Upward) | Moment $M_A = {M_A:.2f}$ {u_moment}")

# 计算剪力 V(x)
V = np.zeros_like(x)
V += R_A
for p1, p2, w in applied_dist:
    mask1 = (x >= p1) & (x < p2)
    V[mask1] -= w * (x[mask1] - p1)
    mask2 = (x >= p2)
    V[mask2] -= w * (p2 - p1)

for pos, P in applied_forces:
    V[x >= pos] -= P

# 计算弯矩 M(x)
M = np.zeros_like(x)
M[0] = M_A
for i in range(1, len(x)):
    M[i] = M[i - 1] + (V[i - 1] + V[i]) / 2 * dx

# 计算挠度 (Deflection)
Slope_EI = np.zeros_like(x)
for i in range(1, len(x)):
    Slope_EI[i] = Slope_EI[i - 1] + (M[i - 1] + M[i]) / 2 * dx

Def_EI = np.zeros_like(x)
for i in range(1, len(x)):
    Def_EI[i] = Def_EI[i - 1] + (Slope_EI[i - 1] + Slope_EI[i]) / 2 * dx

if st.session_state.unit_system == 'SI':
    real_EI = E_display * 1e6 * I_display * 1e-8
    Y_real = Def_EI / real_EI * 1000
else:
    real_EI = E_display * I_display
    Y_real = Def_EI / real_EI


# --- 7. 绘图函数 (完全兼容 Matplotlib) ---
def draw_plots():
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1.5, 1.5, 1.5]})
    fig.patch.set_facecolor('#ffffff')
    plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.05, left=0.1, right=0.95)

    # Plot 1: Load Diagram
    ax0 = axs[0]
    ax0.set_title("Load Diagram", fontsize=10, fontweight='bold', pad=5)
    ax0.set_ylim(-1.5, 1.5)
    ax0.axis('off')

    beam_L = x[-1]
    rect = Rectangle((0, -0.1), beam_L, 0.2, facecolor='gray', edgecolor='black')
    ax0.add_patch(rect)
    ax0.text(0 - 0.5, 0.5, "0", ha='center', fontsize=8, color='red')
    ax0.text(beam_L, 0.5, f"{beam_L}", ha='center', fontsize=8, color='red')

    wall_x = 0
    ax0.plot([wall_x, wall_x], [-0.5, 0.5], 'k-', lw=4)
    for i in np.linspace(-0.5, 0.5, 10):
        ax0.plot([wall_x, wall_x - 0.5], [i, i - 0.1], 'k-', lw=1.5)

    for pos, mag in point_forces:
        scale = 0.5
        ax0.annotate('', xy=(pos, 0.1), xytext=(pos, 0.1 + scale),
                     arrowprops=dict(facecolor='blue', shrink=0.05, width=1.0, headwidth=3, headlength=4))
        ax0.text(pos, 0.1 + scale + 0.1, f"{mag}", ha='center', color='blue', fontsize=8)

    for p1, p2, mag in applied_dist:
        for dp in np.linspace(p1, p2, max(3, int((p2 - p1) / (L / 10)))):
            ax0.annotate('', xy=(dp, 0.1), xytext=(dp, 0.4),
                         arrowprops=dict(facecolor='green', alpha=0.5, width=0.8, headwidth=3, headlength=4,
                                         shrink=0.05))
        ax0.plot([p1, p2], [0.4, 0.4], 'g-', lw=1)
        ax0.text((p1 + p2) / 2, 0.5, f"w={mag}", ha='center', color='green', fontsize=8)

    def plot_diagram(ax, y_data, title, unit, color, fill_color, invert_y=False):
        ax.plot(x, y_data, color=color, lw=1.5)
        ax.fill_between(x, y_data, 0, color=fill_color, alpha=0.3)
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.axhline(0, color='black', lw=0.5)

        if invert_y:
            ax.invert_yaxis()

        idx_max = np.argmax(np.abs(y_data))
        x_max = x[idx_max]
        y_max = y_data[idx_max]

        offset = max(np.abs(y_data)) * 0.15 if max(np.abs(y_data)) != 0 else 1
        if y_max < 0 and not invert_y: offset = -offset
        if y_max > 0 and invert_y: offset = -offset

        ax.annotate(f"{y_max:.2f}", xy=(x_max, y_max), xytext=(x_max, y_max + offset),
                    arrowprops=dict(arrowstyle="->", color=color),
                    color=color, fontweight='bold', fontsize=9, ha='center',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8))

        title_text = f"{title}  [Max: {np.max(y_data):.2f}, Min: {np.min(y_data):.2f}]"
        ax.text(0.02, 1.05, title_text, transform=ax.transAxes, fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plot_diagram(axs[1], V, "Shear Force (V)", f"Force ({u_force})", '#1976d2', '#bbdefb')
    plot_diagram(axs[2], M, "Bending Moment (M)", f"Moment ({u_moment})", '#d32f2f', '#ffcdd2')
    plot_diagram(axs[3], Y_real, "Deflection (w)", f"Disp. ({u_def})", '#388e3c', '#c8e6c9', invert_y=True)
    axs[3].set_xlabel(f"Position x ({u_len})", fontsize=10)

    return fig


st.pyplot(draw_plots())
