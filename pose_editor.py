import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# --- MediaPipeの接続定義 (骨格描画用) ---
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), 
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)
]
# 対応する関節名リスト(標準的な33点)
JOINT_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]
# 名前からインデックスを引く辞書
JOINT_MAP = {name: i for i, name in enumerate(JOINT_NAMES)}

class PoseEditor:
    def __init__(self, csv_path, output_path):
        self.csv_path = csv_path
        self.output_path = output_path
        
        print(f"Loading {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # データの整理
        self.frames = sorted(self.df['frame'].unique())
        self.current_frame_idx = 0
        
        # 描画設定
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.selected_point = None  # (person_id, joint_name)
        self.move_step = 0.05       # 1回押したときの移動量(m)
        
        # イベント接続
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        
        self.text_annot = self.ax.text2D(0.05, 0.95, "", transform=self.ax.transAxes, color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        print("\n=== 操作方法 ===")
        print(" [← / →] : 前/次のフレームへ移動")
        print(" [Click] : 点を選択 (赤く囲まれます)")
        print(" [↑ / ↓] : 選択した点を Y軸方向(奥/手前) に移動")
        print(" [Shift + ←/→] : 選択した点を X軸方向(左右) に移動")
        print(" [Q / E] : 選択した点を Z軸方向(上下) に移動")
        print(" [C] : 選択した人のIDを切り替え (0 <-> 1)")
        print(" [W] : 現在の状態をCSVに保存")
        print("================\n")
        
        self.draw_frame()
        plt.show()

    def get_current_data(self):
        f = self.frames[self.current_frame_idx]
        return self.df[self.df['frame'] == f]

    def draw_frame(self):
        self.ax.cla()
        frame_val = self.frames[self.current_frame_idx]
        subset = self.get_current_data()
        
        # 軸の固定（動きを見やすくするため固定範囲にする）
        self.ax.set_xlim(-2.0, 3.0)
        self.ax.set_ylim(-1.0, 4.0)
        self.ax.set_zlim(0.0, 2.5)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y (Depth)')
        self.ax.set_zlabel('Z (Height)')
        self.ax.set_title(f"Frame: {frame_val} / {self.frames[-1]}")

        # 人ごとにプロット
        pids = subset['person_id'].unique()
        colors = ['blue', 'green', 'orange'] # IDごとの色
        
        self.scatters = {} # pickイベント用に保存

        for pid in pids:
            p_data = subset[subset['person_id'] == pid]
            
            # 座標辞書を作る
            pts = {}
            for _, row in p_data.iterrows():
                pts[row['joint']] = (row['X'], row['Y'], row['Z'])
            
            # 点描画
            xs, ys, zs = [], [], []
            joints = []
            for j, (x, y, z) in pts.items():
                xs.append(x); ys.append(y); zs.append(z)
                joints.append(j)
            
            # 選択中の点は赤枠をつける
            edgecolors = []
            sizes = []
            for j in joints:
                if self.selected_point and self.selected_point[0] == pid and self.selected_point[1] == j:
                    edgecolors.append('red')
                    sizes.append(100)
                else:
                    edgecolors.append('none')
                    sizes.append(20)

            # Scatterプロット (picker=5 はクリック判定の許容範囲)
            sc = self.ax.scatter(xs, ys, zs, c=colors[int(pid) % len(colors)], 
                                 s=sizes, edgecolors=edgecolors, linewidths=2, picker=5, label=f"ID:{pid}")
            
            # イベントハンドラで使うためにデータを保持
            # scオブジェクト -> (pid, joint_list) の対応付け
            self.scatters[sc] = (pid, joints)

            # 骨格線を描画
            for u_idx, v_idx in POSE_CONNECTIONS:
                u_name = JOINT_NAMES[u_idx] if u_idx < len(JOINT_NAMES) else ""
                v_name = JOINT_NAMES[v_idx] if v_idx < len(JOINT_NAMES) else ""
                
                if u_name in pts and v_name in pts:
                    p1 = pts[u_name]
                    p2 = pts[v_name]
                    self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=colors[int(pid) % len(colors)], alpha=0.5)

        self.ax.legend()
        self.fig.canvas.draw()

    def on_pick(self, event):
        # 点をクリックしたとき
        sc = event.artist
        if sc in self.scatters:
            pid, joints = self.scatters[sc]
            ind = event.ind[0] # クリックした点のインデックス
            joint_name = joints[ind]
            
            self.selected_point = (pid, joint_name)
            print(f"Selected: ID {pid} - {joint_name}")
            self.draw_frame()

    def on_hover(self, event):
        # マウスホバー処理 (簡易実装: マウス位置に近い点を投影して探すのは重いので、pickerの情報を利用したほうがいいが、
        # 3DプロットでのホバーはMatplotlibでは難しい。ここでは「選択中の情報」を常に表示する形式にする)
        if self.selected_point:
            f = self.frames[self.current_frame_idx]
            pid, jname = self.selected_point
            # 現在の値を取得
            row = self.df[(self.df['frame'] == f) & (self.df['person_id'] == pid) & (self.df['joint'] == jname)]
            if not row.empty:
                x, y, z = row.iloc[0][['X', 'Y', 'Z']]
                self.text_annot.set_text(f"Selected: ID{pid} {jname}\n({x:.2f}, {y:.2f}, {z:.2f})")
            else:
                self.text_annot.set_text("Selected point missing in this frame")
        else:
            self.text_annot.set_text("Click a point to select")
        
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        key = event.key
        f = self.frames[self.current_frame_idx]

        # 1. フレーム移動
        if key == 'right':
            if self.current_frame_idx < len(self.frames) - 1:
                self.current_frame_idx += 1
                self.draw_frame()
            return
        elif key == 'left':
            if self.current_frame_idx > 0:
                self.current_frame_idx -= 1
                self.draw_frame()
            return

        # 2. 保存
        if key == 'w':
            print(f"Saving to {self.output_path}...")
            self.df.to_csv(self.output_path, index=False)
            print("Saved!")
            return

        # これ以降は「点が選択されている」場合のみ有効
        if not self.selected_point:
            return

        pid, jname = self.selected_point
        
        # 対象の行を特定
        mask = (self.df['frame'] == f) & (self.df['person_id'] == pid) & (self.df['joint'] == jname)
        if not mask.any(): return
        idx = self.df.index[mask][0]

        # 3. ID変更 (Toggle 0 <-> 1)
        if key == 'c':
            new_id = 1 if pid == 0 else 0
            # ★重要: そのフレームの「その人の全関節」のIDを変える（手首だけID変わるとバグるため）
            mask_person = (self.df['frame'] == f) & (self.df['person_id'] == pid)
            self.df.loc[mask_person, 'person_id'] = new_id
            print(f"Frame {f}: Swapped ID {pid} -> {new_id}")
            self.selected_point = (new_id, jname) # 選択状態も追従
            self.draw_frame()
            return

        # 4. 座標移動
        # X軸: Shift + 左右
        if key == 'shift+right':
            self.df.at[idx, 'X'] += self.move_step
        elif key == 'shift+left':
            self.df.at[idx, 'X'] -= self.move_step
        
        # Y軸 (奥行き): 上下
        elif key == 'up':
            self.df.at[idx, 'Y'] += self.move_step
        elif key == 'down':
            self.df.at[idx, 'Y'] -= self.move_step
        
        # Z軸 (高さ): Q/E
        elif key == 'e':
            self.df.at[idx, 'Z'] += self.move_step
        elif key == 'q':
            self.df.at[idx, 'Z'] -= self.move_step

        # 再描画
        self.draw_frame()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Input CSV")
    parser.add_argument("--output", "-o", type=str, default="kakunin_edited.csv", help="Output CSV")
    args = parser.parse_args()

    PoseEditor(args.input, args.output)

if __name__ == "__main__":
    main()