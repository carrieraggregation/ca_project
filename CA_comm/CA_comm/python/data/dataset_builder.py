import numpy as np
import os
import pandas as pd
from typing import Tuple

def build_sequences(df: pd.DataFrame, history: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    DataFrame에서 UE별 시계열 CQI 데이터를 뽑아, LSTM 학습용 X, y 배열을 생성합니다.

    Inputs:
      df      : pandas DataFrame, 컬럼에 'CQI_CC1'..'CQI_CCn', 'X','Y','Z','Distance' 포함
      history : 과거 TTI 개수 (슬라이딩 윈도우 크기)

    Returns:
      X : numpy array of shape (num_samples, history * numCC + num_static_features)
      y : numpy array of shape (num_samples, numCC)

    예: history=10, numCC=5, static=4 -> 각 X row = 10*5 + 4 = 54 dim
    """
    # CQI 컬럼 검색 및 정렬 (예: 'CQI_CC1', 'CQI_CC2', ...)
    cqi_cols = sorted(
        [c for c in df.columns if c.startswith('CQI_CC')],
        key=lambda x: int(x.split('CQI_CC')[-1])
    )
    numCC = len(cqi_cols)
    # 고정 feature (위치, 거리)
    static_cols = ['X', 'Y', 'Z', 'Distance']

    X_list = []
    y_list = []
    # Seed, UE 단위로 그룹핑
    for (_, _), group in df.groupby(['Seed', 'UE']):
        grp = group.reset_index(drop=True)
        if len(grp) <= history:
            continue
        for idx in range(history, len(grp)):
            # 과거 CQI 히스토리
            past = grp.loc[idx-history:idx-1, cqi_cols].values  # (history, numCC)
            # 고정 feature
            static = grp.loc[idx, static_cols].values           # (4,)
            # 피처 결합
            X_feat = np.hstack((past.flatten(), static))        # (history*numCC + len(static_cols),)
            # 목표: 현재 TTI CQI
            y_feat = grp.loc[idx, cqi_cols].values              # (numCC,)
            X_list.append(X_feat)
            y_list.append(y_feat)

    X = np.vstack(X_list) if X_list else np.empty((0, history*numCC + len(static_cols)))
    y = np.vstack(y_list) if y_list else np.empty((0, numCC))
    return X, y


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build LSTM dataset from TTI logs')
    parser.add_argument(
        '--csv', type=str, default=None,
        help='입력 CSV 파일 경로 (기본: matlab/logs/tti_logs.csv)'
    )
    parser.add_argument(
        '--history', type=int, default=10,
        help='과거 TTI 개수'
    )
    args = parser.parse_args()

    # 1) 경로 설정
    script_dir   = os.path.dirname(os.path.abspath(__file__))          # .../python/data
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))  # .../python
    logs_dir     = os.path.join(project_root, 'matlab', 'logs')         # .../matlab/logs

    # 2) CSV 파일 경로 결정
    if args.csv and os.path.isabs(args.csv):
        csv_path = args.csv
    else:
        csv_file = args.csv or 'tti_logs.csv'
        csv_path = os.path.join(logs_dir, csv_file)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # 3) 데이터 로드 및 시퀀스 생성
    df = pd.read_csv(csv_path)
    X, y = build_sequences(df, args.history)
    print(f'Generated sequences: X shape = {X.shape}, y shape = {y.shape}')

    # 4) NPY 저장 경로 결정
    X_path = os.path.join(logs_dir, 'X.npy')
    y_path = os.path.join(logs_dir, 'y.npy')

    np.save(X_path, X)
    np.save(y_path, y)
    print(f"Saved X to {X_path}")
    print(f"Saved y to {y_path}")