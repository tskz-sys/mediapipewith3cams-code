import cv2
from ultralytics import YOLO
import numpy as np

def run_yolo_tracking_br_setup(video_paths, output_paths=None, model_path='yolov8n.pt'):
    """
    BR (Back/Right 等) 構成のカメラに対してYOLOによるID付きトラッキングを行うコード
    
    Args:
        video_paths (dict): {'B': 'path_to_video_B', 'R': 'path_to_video_R'} の形式
        output_paths (dict): 保存先のパス {'B': 'output_B.mp4', 'R': 'output_R.mp4'} (Noneの場合は保存なし)
        model_path (str): YOLOモデルのパス
    """
    
    # YOLOモデルのロード
    # model = YOLO(model_path) 
    # 必要であればカスタムモデルを使用してください
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)

    # ビデオキャプチャの初期化
    caps = {}
    writers = {}
    
    for cam_id, path in video_paths.items():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Could not open video source for Camera {cam_id}: {path}")
            return
        caps[cam_id] = cap
        
        # 保存設定（必要な場合）
        if output_paths:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writers[cam_id] = cv2.VideoWriter(output_paths[cam_id], fourcc, fps, (width, height))

    print("Starting detection loop for Camera B and R. Press 'q' to exit.")

    while True:
        frames = {}
        ret_status = {}

        # 各カメラからフレーム読み込み
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            ret_status[cam_id] = ret
            if ret:
                frames[cam_id] = frame

        # どちらかのカメラが終了したらループを抜ける（用途に合わせて調整してください）
        if not all(ret_status.values()):
            print("End of video stream detected.")
            break

        # 各フレームに対して推論と描画
        for cam_id, frame in frames.items():
            # YOLO Tracking (persist=TrueでIDを維持)
            # classes=0 は 'person' のみを対象とする場合（適宜変更してください）
            results = model.track(frame, persist=True, verbose=False, classes=0)

            # 結果の描画
            annotated_frame = frame.copy()
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    
                    # バウンディングボックス描画
                    color = (0, 255, 0) if cam_id == 'B' else (0, 0, 255) # Bは緑、Rは赤などで区別
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # IDとラベル描画
                    label = f"ID: {track_id} ({cam_id})"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 画面表示用のリサイズ（必要に応じて）
            display_frame = cv2.resize(annotated_frame, (640, 360))
            cv2.imshow(f'Camera {cam_id}', display_frame)

            # 保存
            if output_paths and cam_id in writers:
                writers[cam_id].write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソース解放
    for cap in caps.values():
        cap.release()
    for writer in writers.values():
        writer.release()
    cv2.destroyAllWindows()
    print("Processing complete.")

# --- 実行例 ---
if __name__ == "__main__":
    # 入力動画パス（適宜書き換えてください）
    # 'B' と 'R' をキーとして使用します（LR問題→BR問題への対応）
    input_videos = {
        'B': 'path/to/camera_back.mp4',  # Back / Base Camera
        'R': 'path/to/camera_right.mp4'   # Right / Ref Camera
    }
    
    # 出力パス（保存しない場合はNone）
    output_videos = {
        'B': 'output_B.mp4',
        'R': 'output_R.mp4'
    }

    # テスト実行のため、パスが存在する場合のみ実行などのチェックを入れても良い
    # run_yolo_tracking_br_setup(input_videos, output_videos)