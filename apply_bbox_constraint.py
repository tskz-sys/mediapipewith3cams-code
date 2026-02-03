import pandas as pd
import numpy as np
import cv2
import argparse
import os

def load_calibration(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    def get_k(k_new, k_old): return d[k_new] if k_new in d else d[k_old]
    
    cams = []
    for i in range(1, 4):
        K = get_k(f"K{i}", f"cam_matrix{i}")
        D = get_k(f"dist{i}", f"dist_coeffs{i}")
        if i == 1:
            R, t = d["R1"], d["T1"] if "T1" in d else d["t1"]
            R_ext, t_ext = R.T, -R.T @ t.reshape(3,1)
        elif i == 2:
            R_ext, t_ext = np.eye(3), np.zeros((3, 1))
        else:
            R, t = d["R3"], d["T3"] if "T3" in d else d["t3"]
            R_ext, t_ext = R.T, -R.T @ t.reshape(3,1)
        cams.append({"K": K, "D": D, "R": R_ext, "t": t_ext})
    return cams

def project_point(pt_3d, cam):
    if np.any(np.isnan(pt_3d)): return None
    pt_3d = pt_3d.reshape(1, 1, 3)
    rvec, _ = cv2.Rodrigues(cam['R'])
    tvec = cam['t']
    img_pts, _ = cv2.projectPoints(pt_3d, rvec, tvec, cam['K'], cam['D'])
    return img_pts[0][0] # [x, y]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_3d", required=True, help="Smoothed 3D CSV")
    parser.add_argument("--input_bbox", required=True, help="BBox Log CSV from detection")
    parser.add_argument("--calib", required=True, help="Calibration NPZ")
    parser.add_argument("--output", required=True, help="Constrained CSV")
    parser.add_argument("--margin", type=float, default=1.1, help="Allow slight margin (e.g. 1.1x bbox size)")
    args = parser.parse_args()

    print("Loading data...")
    df_3d = pd.read_csv(args.input_3d)
    df_bbox = pd.read_csv(args.input_bbox)
    cams = load_calibration(args.calib)
    
    # 高速化のため辞書化
    # bbox_map[frame][pid][cam_idx] = [x1, y1, x2, y2]
    bbox_map = {}
    for _, row in df_bbox.iterrows():
        f = int(row['frame'])
        pid = int(row['person_id'])
        cid = int(row['cam_idx'])
        if f not in bbox_map: bbox_map[f] = {}
        if pid not in bbox_map[f]: bbox_map[f][pid] = {}
        bbox_map[f][pid][cid] = [row['x1'], row['y1'], row['x2'], row['y2']]

    print("Checking constraints...")
    
    valid_rows = []
    out_of_bounds_count = 0
    total_points = len(df_3d)

    for _, row in df_3d.iterrows():
        f = int(row['frame'])
        pid = int(row['person_id'])
        
        # 3D座標
        pt_3d = np.array([row['X'], row['Y'], row['Z']])
        
        # このフレーム・この人のBBox情報があるか？
        if f in bbox_map and pid in bbox_map[f]:
            cam_boxes = bbox_map[f][pid]
            
            is_valid = True
            
            # BBoxが存在する全てのカメラについてチェック
            for cid, box in cam_boxes.items():
                pt_2d = project_point(pt_3d, cams[cid])
                
                if pt_2d is None: continue # 投影不能
                
                # BBox範囲チェック (マージン付き)
                x1, y1, x2, y2 = box
                w, h = x2-x1, y2-y1
                cx, cy = (x1+x2)/2, (y1+y2)/2
                
                # マージン適用後の範囲
                w_m, h_m = w * args.margin, h * args.margin
                x1_m, x2_m = cx - w_m/2, cx + w_m/2
                y1_m, y2_m = cy - h_m/2, cy + h_m/2
                
                px, py = pt_2d
                if not (x1_m <= px <= x2_m and y1_m <= py <= y2_m):
                    is_valid = False
                    break # 1つでもカメラ範囲外ならNGとする
            
            if is_valid:
                valid_rows.append(row)
            else:
                out_of_bounds_count += 1
                # NGの場合は行を追加しない（削除する）
                # もし「元の値に戻す」処理がしたい場合は、ここに未処理CSVの値を参照するロジックが必要
        else:
            # BBox情報がない（見失っている）期間は、スムージング結果を信じるしかない
            valid_rows.append(row)

    print(f"Finished. Removed {out_of_bounds_count} points out of {total_points} ({out_of_bounds_count/total_points*100:.1f}%)")
    
    out_df = pd.DataFrame(valid_rows)
    out_df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()