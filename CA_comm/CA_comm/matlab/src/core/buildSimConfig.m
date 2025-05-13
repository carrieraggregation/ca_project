function simConfig = buildSimConfig()
% buildSimConfig  시뮬레이션 전반에 사용될 파라미터 정의
% 출력:
%   simConfig – struct with fields:
%       .numTTI, .numUE, .numCC
%       .areaLength, .gNBHeight, .ueHeight, .buildingHeight, .streetWidth
%       .baseCarrierFreq, .scs_kHz, .SubcarrierSpacing, .nRB, .nSC, .nSymbols
%       .txPower_dBm, .noiseFigure_dB, .Velocity

    % 1) 반복 및 TTI 설정
    simConfig.numTTI = 20;            % 총 TTI 수
    simConfig.numUE  = 20;            % 전체 UE 수
    simConfig.numCC  = 5;             % Component Carrier 수

    % 2) 환경 파라미터
    simConfig.areaLength     = 300;   % 시뮬레이션 영역 (m)
    simConfig.gNBHeight      = 30;    % gNB 높이 (m)
    simConfig.ueHeight       = 1.5;   % UE 높이 (m)
    simConfig.buildingHeight = 20;    % 평균 건물 높이 (m)
    simConfig.streetWidth    = 20;    % 거리 폭 (m)

    % 3) 주파수 및 자원 블록
    simConfig.baseCarrierFreq   = 3.6e9;  % 중심 주파수 (Hz)
    simConfig.scs_kHz           = 15;     % 서브캐리어 간격 (kHz)
    simConfig.SubcarrierSpacing = simConfig.scs_kHz * 1e3;  % (Hz)
    simConfig.nRB   = 106;      % CC당 RB 개수
    simConfig.chunkSize = 5;    % 한 번에 5개 cc 할당
    simConfig.nSC   = 12;       % RB당 서브캐리어 개수
    simConfig.nSymbols = 14;    % 1 TTI당 OFDM 심볼 수

    % 4) 전송/수신 파라미터
    simConfig.txPower_dBm    = 23;  % 기지국 송신 전력 (dBm)
    simConfig.noiseFigure_dB = 7;   % 수신단 잡음 지수 (dB)

    % 5) UE 이동성
    simConfig.Velocity  = 3;                              % UE 속도 (m/s)
    simConfig.BW        = 100 * ones(1, simConfig.numCC); % 각 CC 대역폭 (MHz)
    simConfig.gamma     = 5.0;                            % 제어 오버헤드 계수 (Mbps 감소량 per SCell)
end
