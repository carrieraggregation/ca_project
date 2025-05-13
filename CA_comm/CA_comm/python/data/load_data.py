import scipy.io as sio
import pandas as pd
import os
import argparse

def load_mat_logs(mat_path: str) -> pd.DataFrame:
    """
    .mat 파일에서 MATLAB 시뮬레이션 로그(allLogData)를 읽어
    pandas DataFrame으로 반환
    """
    mat = sio.loadmat(mat_path)
    data = mat['allLogData']  # shape = (numTTI*numUE, numCols)

    # buildHeaders 순서와 일치하도록 칼럼명 정의
    cols = [
        'Seed', 'UE', 'X', 'Y', 'Z', 'Distance', 'AvgCQI',
        *[f'CC{i}' for i in range(1, 6)],
        *[f'CQI_CC{i}' for i in range(1, 6)],
        *[f'Thr_CC{i}_Mbps' for i in range(1, 6)],
        'TotalThr_Mbps'
    ]

    df = pd.DataFrame(data, columns=cols)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load MATLAB TTI logs from matlab/logs and save as CSV in the same folder'
    )
    parser.add_argument(
        '--mat', type=str, default='tti_logs_seed101.mat',
        help='.mat 파일명 (matlab/logs 폴더 내)'
    )
    parser.add_argument(
        '--out', type=str, default=None,
        help='출력 CSV 파일명. 지정하지 않으면 .mat 파일명으로 .csv 저장'
    )
    args = parser.parse_args()

    # script_dir: python/data
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    logs_dir     = os.path.join(project_root, 'matlab', 'logs')

    # mat 파일 경로 설정
    mat_path = args.mat if os.path.isabs(args.mat) else os.path.join(logs_dir, args.mat)
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    # out 파일명 결정
    base_name = os.path.splitext(os.path.basename(mat_path))[0]
    out_name  = args.out if args.out else f"{base_name}.csv"
    out_path  = out_name if os.path.isabs(out_name) else os.path.join(logs_dir, out_name)

    # 로드 후 CSV 저장
    df = load_mat_logs(mat_path)
    print(df.head())
    df.to_csv(out_path, index=False)
    print(f"Saved DataFrame to {out_path}")
