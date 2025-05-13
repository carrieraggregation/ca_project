function scenario = generateScenario(simConfig, seedId)
% generateScenario  한 시드에 대한 시나리오(UE 배치, 채널 등) 생성
% 입력:
%   simConfig – 시뮬레이션 설정(struct)
%   seedId    – RNG 시드 (scalar)
% 출력:
%   scenario – struct with fields:
%     .seedId, .numUE, .numCC, .gNBPos, .UEPos, .CC (1×numCC struct array)

% 1) 랜덤 시드 고정
if nargin > 1
    rng(seedId);
else
    rng('default');
end
scenario.seedId = seedId;
% 시나리오에 numUE, numCC 추가
scenario.numUE = simConfig.numUE;
scenario.numCC = simConfig.numCC;

% 2) gNB 및 UE 위치
scenario.gNBPos = [0; 0; simConfig.gNBHeight];
% UE 위치: X,Y는 영역(-L/2, L/2), Z는 ueHeight
X = (rand(1,simConfig.numUE) - 0.5) * simConfig.areaLength;
Y = (rand(1,simConfig.numUE) - 0.5) * simConfig.areaLength;
Z = ones(1,simConfig.numUE) * simConfig.ueHeight;
scenario.UEPos = [X; Y; Z];

% 3) PathLoss 설정
plConf = nrPathLossConfig;
plConf.Scenario       = 'UMa';
plConf.BuildingHeight = simConfig.buildingHeight;
plConf.StreetWidth    = simConfig.streetWidth;
losMatrix = false(1,simConfig.numUE);

% 4) CC별 정보 및 TDL 채널 생성
scenario.CC = repmat(struct( ...
    'CenterFreq',       [], ...
    'PathLossPerUE_dB', [], ...
    'ShadowingPerUE_dB',[], ...
    'TDLchan',          []), 1, simConfig.numCC);

for ccIdx = 1:simConfig.numCC
    % 중심 주파수
    cf = simConfig.baseCarrierFreq + (ccIdx-1)*20e6;
    
    % 경로손실 & 섀도잉
    [pl_dB, shadowStd] = nrPathLoss(plConf, cf, losMatrix, scenario.gNBPos, scenario.UEPos);
    shadowing = randn(1,simConfig.numUE) .* shadowStd;

    % TDL 채널 객체 (2×2 MIMO)
    fs = simConfig.nSC * simConfig.SubcarrierSpacing;  % sample rate
    fd = simConfig.Velocity * cf / physconst('LightSpeed');

    % CC별로 UE × 1 cell array 채널 객체
    scenario.CC(ccIdx).TDLchan = cell(1, simConfig.numUE);
    for ueIdx = 1:simConfig.numUE
        scenario.CC(ccIdx).TDLchan{ueIdx} = nrTDLChannel( ...
            'SampleRate',          fs, ...
            'DelayProfile',        'TDL-C', ...
            'NumTransmitAntennas', 2, ...
            'NumReceiveAntennas',  2, ...
            'MaximumDopplerShift', fd);
    end

    % 저장
    scenario.CC(ccIdx).CenterFreq        = cf;
    scenario.CC(ccIdx).PathLossPerUE_dB  = pl_dB;
    scenario.CC(ccIdx).ShadowingPerUE_dB = shadowing;
    % scenario.CC(ccIdx).TDLchan already set per UE above
end
end
