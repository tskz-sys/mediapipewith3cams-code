import pandas as pd
import numpy as np
import argparse

def main():
    # ファイル設定（必要に応じて書き換えてください）
    CSV_FILE = "./csv/kakunin_constrained.csv"
    TARGET_FRAME_START = 0
    TARGET_FRAME_END = 420
    JUMP_THRESHOLD = 0.3  # 1フレームで0.3m以上動いたら「異常」とみなす

    print(f"Checking {CSV_FILE} (Frame {TARGET_FRAME_START}-{TARGET_FRAME_END})...")
    
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} が見つかりません。")
        return

    # 指定範囲に絞り込み
    df = df[(df['frame'] >= TARGET_FRAME_START) & (df['frame'] <= TARGET_FRAME_END)]
    
    # 代表として「右足首(right_ankle)」または「腰(right_hip)」を見る
    # ※足は速く動くので、少し閾値甘めでもOK。腰は安定しやすい。
    target_joint = "right_ankle"
    df_joint = df[df['joint'] == target_joint].copy()

    # IDごとに処理
    person_ids = sorted(df_joint['person_id'].unique())
    
    if len(person_ids) == 0:
        print("指定範囲にデータがありません。")
        return

    print(f"\n=== 安定性チェック (部位: {target_joint}) ===")
    
    all_stable = True

    for pid in person_ids:
        print(f"\nUser ID: {pid}")
        print(f"{'Frame':<6} | {'X':<7} | {'Y':<7} | {'Z':<7} | {'Move(m)':<8} | {'Status'}")
        print("-" * 60)

        # IDごとにフレーム順に並べ替え
        sub = df_joint[df_joint['person_id'] == pid].sort_values('frame')
        
        prev_row = None
        max_jump = 0.0
        
        for _, row in sub.iterrows():
            move = 0.0
            mark = "OK"
            
            if prev_row is not None:
                # フレームが連続しているか確認
                frame_diff = row['frame'] - prev_row['frame']
                
                # 距離計算
                dist = np.sqrt((row['X']-prev_row['X'])**2 + (row['Y']-prev_row['Y'])**2 + (row['Z']-prev_row['Z'])**2)
                
                # 1フレームあたりの移動量に換算（欠損があった場合のため）
                move = dist / frame_diff if frame_diff > 0 else 0
                
                if move > JUMP_THRESHOLD:
                    mark = "JUMP!!"
                    all_stable = False
                
                if move > max_jump:
                    max_jump = move

            print(f"{int(row['frame']):<6} | {row['X']:.3f} | {row['Y']:.3f} | {row['Z']:.3f} | {move:.4f}   | {mark}")
            prev_row = row
            
        print(f"-> Max Jump for ID {pid}: {max_jump:.4f} m/frame")

    print("\n" + "="*30)
    if all_stable:
        print("【判定: 安定 (Stable)】")
        print("CSVデータに異常な飛びはありません。")
        print("映像のズレは「キャリブレーション表示上の誤差」であり、データ解析には問題ありません。")
        print("修正の必要はありません。")
    else:
        print("【判定: 不安定 (Unstable)】")
        print("データ自体が瞬間移動しています。")
        print("IDスイッチング（別人と入れ替わり）が発生している可能性があります。")
        print("fix_pose_csv_adaptive.py の実行を推奨します。")
    print("="*30)

if __name__ == "__main__":
    main()