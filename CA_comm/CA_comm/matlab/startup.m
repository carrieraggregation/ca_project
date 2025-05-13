% matlab/startup.m — 프로젝트 초기화 스크립트

% 1) 프로젝트 루트 경로 설정
projectRoot = fileparts(mfilename('fullpath'));

% 2) MATLAB 소스 폴더 추가
addpath(fullfile(projectRoot, 'src', 'core'));
addpath(fullfile(projectRoot, 'src', 'scheduler'));
addpath(fullfile(projectRoot, 'src', 'phy'));
addpath(fullfile(projectRoot, 'src', 'utils'));

% 3) 데이터·로그 폴더 추가 (선택)
addpath(fullfile(projectRoot, 'logs'));

% 4) Python 및 ZeroMQ 환경 설정
pe = pyenv( ...
    'Version',       'C:/Users/LGPC/AppData/Local/Programs/Python/Python310/python.exe', ...  % pyzmq 설치된 Python 경로
    'ExecutionMode', 'OutOfProcess' ...
    );
fprintf('Using Python %s (ExecutionMode=%s)\n', pe.Version, pe.ExecutionMode);

% 5) (선택) Python 모듈 경로 추가
if count(py.sys.path, projectRoot) == 0
    insert(py.sys.path, int32(0), projectRoot);
end
