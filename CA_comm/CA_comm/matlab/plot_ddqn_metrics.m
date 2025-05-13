% ───────────────────────────────────────────────────────
% plot_ddqn_metrics.m
% MATLAB에서 Python metrics.py 모듈을 불러오기 위해
% metrics.py가 있는 폴더를 Python path에 추가하는 코드
% ───────────────────────────────────────────────────────

% 1) 이 스크립트 파일이 있는 폴더 경로 (예: .../CA_comm/matlab)
scriptDir = fileparts( mfilename('fullpath') );

% 2) 프로젝트 루트 폴더 경로 (예: .../CA_comm)
projectDir = fileparts( scriptDir );

% 3) metrics.py가 위치한 Python/scripts 폴더 경로
%    → CA_comm/python/scripts/metrics.py
pythonScriptsDir = fullfile( projectDir, 'python', 'scripts' );

% 4) MATLAB 내 Python 검색 경로(py.sys.path)에 없으면 맨 앞에 추가
if count(py.sys.path, pythonScriptsDir) == 0
    insert(py.sys.path, int32(0), pythonScriptsDir);
end

% 5) 이제 metrics 모듈을 import
metrics = py.importlib.import_module('metrics');

rewards_py = metrics.get_metrics();
rewards = double(rewards_py);

% 6) 플롯
figure;
plot(rewards, '-o','LineWidth',1.5);
xlabel('Train Step (TTI count)');
ylabel('Reward');
title('DDQN Reward History');
grid on;
