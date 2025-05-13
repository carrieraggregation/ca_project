function [ttiResult, stateUpdate, logOut] = simulateTTI(scenario, ttiHistory, simConfig, tti)
% simulateTTI  한 TTI 동안 CQI → 스케줄링 → PHY 시뮬 → 로그 생성
% 입력:
%   scenario   – generateScenario에서 생성된 struct
%   ttiHistory – 이전 TTI까지의 history struct array
%   simConfig  – 시뮬레이션 설정 struct
% 출력:
%   ttiResult  – 1×numUE struct array with fields UE, Position, Distance,
%                CCMask, PerCC (struct array), AvgCQI, TotalThroughput
%   stateUpdate– struct with lastThroughput (numUE×numCC) and avgThroughput
%   logOut     – struct with headers (cellstr) and data (numUE×N double)

%% 0) CQI 테이블 & 임계 SNR (표준 3GPP NR CQI Table #1 사용)
cqiTables    = nrCQITables();
cqiTbl       = cqiTables.CQITable1;                     % 객체 이름을 cqiTbl로 통일
snrThresh_dB = 10*log10(2.^cqiTbl.SpectralEfficiency - 1);  % [15×1] 벡터


%% 1) PF Metric 계산
numUE = simConfig.numUE;
numCC = simConfig.numCC;
if isempty(ttiHistory) || ~isfield(ttiHistory(end),'lastThroughput') || isempty(ttiHistory(end).lastThroughput)
    pfMetric = ones(numUE, numCC);
else
    pfMetric = computePFMetric(ttiHistory);
end

%% 2) CC 활성화 및 RB 할당 (Python DDQN infer 연동)
% 이전 TTI의 CC mask 복원
if isempty(ttiHistory)
    prevMask = false(numUE, numCC);
else
    prevMask = ttiHistory(end).CCMask;
end

% 1) PF metric과 이전 CC mask를 flatten
pfVec       = pfMetric(:)';                 % 1×(numUE×numCC)
prevMaskVec = prevMask(:)';                 % 1×(numUE×numCC)

% 2) Python 서버에 infer 요청 → maskIdx (스칼라) + Q-values
[maskIdx, qValues] = sendInferenceToPython(pfVec, prevMaskVec);

% 3) maskIdx → UE×CC 논리 행렬 복원
binMat = de2bi(maskIdx, numCC, 'left-msb');  % 1×numCC → 1행
binMat(1) = 1;                        % PCell(bit1) 항상 1로 고정
maskIdx = bi2de(binMat, 'left-msb');
ccMask = repmat(logical(binMat), numUE, 1);
% 4) RB 할당
rbAlloc = pfScheduler(pfMetric, ccMask, simConfig.nRB, simConfig.chunkSize);  % [UE×CC×RB]
allocMat= sum(rbAlloc, 3);  % [UE×CC]

%% 3) 로그용 헤더 및 버퍼
headers      = buildHeaders(numCC);
nCols        = numel(headers);
logData      = nan(numUE, nCols);
ttiResult(1,numUE) = struct('UE',[],'Position',[],'Distance',[], ...
    'CCMask',[],'PerCC',[],'AvgCQI',[],'TotalThroughput',[]);
lastThr      = zeros(numUE, numCC);

%% 4) UE별 처리
for ue = 1:numUE
    pos      = scenario.UEPos(:, ue);
    distance = norm(pos - scenario.gNBPos);
    cqiList  = nan(1, numCC);
    perCCThr = zeros(1, numCC);
    perCCInfo(1,numCC) = struct('CC',[],'Freq',[],'SNR',[],'CQI',[],'SE',[],'Thr_bps',[]);

    for cc = 1:numCC
        % 4.1) 평균 SNR 계산
        ccStruct = scenario.CC(cc);
        pl_dB     = ccStruct.PathLossPerUE_dB(ue);
        sh_dB     = ccStruct.ShadowingPerUE_dB(ue);
        noise_dBm = -174 + 10*log10(simConfig.nRB*simConfig.SubcarrierSpacing*simConfig.nSC) + simConfig.noiseFigure_dB;
        snrAvg    = simConfig.txPower_dBm - pl_dB - sh_dB - noise_dBm;

        % 4.2) time-varying small-scale fading via TDL channel per UE/CC
        rbCnt   = allocMat(ue,cc);
        chanObj = scenario.CC(cc).TDLchan{ue};
        % Determine number of transmit antennas
        numTx = chanObj.NumTransmitAntennas;
        % OFDM symbol length (subcarriers)
        numSamp = simConfig.nSC;

        % === 교체: OFDM pilot 기반 채널 샘플링 ===
        Nsc   = simConfig.nSC;                       % 서브캐리어 수
        Nsym  = simConfig.nSymbols;                  % OFDM 심볼 수
        CP    = Nsc/4;                               % CP 길이
        % 1) pilot 심볼: 모든 서브캐리어에 '1'
        pilotSym = ones(numTx, Nsc, Nsym);
        % 2) IFFT + CP 붙이기
        txOfdm = [];
        for s = 1:Nsym
            offt = ifft(pilotSym(:,:,s), Nsc, 2) * sqrt(Nsc);
            cp   = offt(:, end-CP+1:end);
            txOfdm = [txOfdm, cp, offt];             % [numTx × (CP+Nsc)]
        end
        % 3) TDL 채널 통과
        rxOfdm = chanObj(txOfdm.');                  % [time×numRx]
        rxOfdm = rxOfdm.';                           % [numRx×time]
        % 4) FFT 후 응답 평균
        Hsum = 0;
        idx  = 1;
        for s = 1:Nsym
            seg = rxOfdm(:, idx+CP : idx+CP+Nsc-1);  % CP 제거
            Hk  = fft(seg, Nsc, 2) / sqrt(Nsc);      % [numRx×Nsc]
            Hsum = Hsum + mean(abs(Hk), 'all');      % magnitude 평균
            idx  = idx + CP + Nsc;
        end
        Hmean = Hsum / Nsym;
        fad_dB = 20*log10(Hmean);                   % dB 스케일 fading
        % =======================================


        % Apply fading to average SNR
        snrF = snrAvg + fad_dB;

        % 4.3) 테이블 기반 CQI 결정
        % → snrF 이상인 마지막 인덱스 찾기
        idx = find(snrF >= snrThresh_dB, 1, 'last');
        if isempty(idx)
            idx = 1;   % SNR이 최저 문턱 이하인 경우 CQI=1
        end
        cqiList(cc) = cqiTbl.CQIIndex(idx);  % 일반적으로 cqiTbl.CQIIndex == 1:1:15

        % 4.4) PHY 시뮬레이션
        if rbCnt > 0
            % CQI 테이블에서 가져온 문자열
            modVal = cqiTbl.Modulation{idx};
            % "Out of Range" 인 경우는 처리량 0
            if strcmpi(modVal, 'Out of Range')
                thr_bps = 0;
            else
                % 정상 modulation만 PHY 시뮬 실행
                modStr = char(modVal);
                [~, thr_bps] = simulateLinkPHY(snrF, modStr, simConfig, rbCnt, chanObj);
            end
        else
            thr_bps = 0;
        end
        perCCThr(cc) = thr_bps;
        lastThr(ue,cc)= thr_bps;

        % 4.5) CC 상세 정보
        perCCInfo(cc).CC      = cc;
        perCCInfo(cc).Freq    = ccStruct.CenterFreq;
        perCCInfo(cc).SNR     = snrF;
        perCCInfo(cc).CQI     = idx;
        perCCInfo(cc).SE      = cqiTbl.SpectralEfficiency(idx);
        perCCInfo(cc).Thr_bps = thr_bps;
    end

    % 5) UE 종합 저장
    avgCQI = mean(cqiList);
    totalThr = sum(perCCThr);
    ttiResult(ue).UE             = ue;
    ttiResult(ue).Position       = pos;
    ttiResult(ue).Distance       = distance;
    ttiResult(ue).CCMask         = ccMask(ue,:);
    ttiResult(ue).PerCC          = perCCInfo;
    ttiResult(ue).AvgCQI         = avgCQI;
    ttiResult(ue).TotalThroughput= totalThr/1e6;

    % 6) 로그 기록 (명시적 인덱스 방식)
    row = nan(1, numel(headers));
    col = 1;
    row(col) = scenario.seedId;         col = col + 1;
    row(col) = ue;                      col = col + 1;
    row(col:col+2) = pos.';             col = col + 3;
    row(col) = distance;                col = col + 1;
    row(col) = avgCQI;                  col = col + 1;

    % CC mask (1~numCC)
    row(col:col+numCC-1) = ccMask(ue,:); col = col + numCC;
    % CQI per CC
    row(col:col+numCC-1) = cqiList;      col = col + numCC;
    % Throughput per CC (Mbps)
    row(col:col+numCC-1) = perCCThr/1e6; col = col + numCC;

    % Total Throughput (Mbps)
    row(col) = sum(perCCThr)/1e6;

    logData(ue,:) = row;

end

%% 7) stateUpdate 계산
if ~isempty(ttiHistory) && isfield(ttiHistory(end),'avgThroughput') && isequal(size(ttiHistory(end).avgThroughput), size(lastThr))
    stateUpdate.avgThroughput = 0.9*ttiHistory(end).avgThroughput + 0.1*lastThr;
else
    stateUpdate.avgThroughput = lastThr;
end
stateUpdate.lastThroughput = lastThr;
% --- 추가: 직전 TTI의 CC 활성화 마스크도 저장
stateUpdate.CCMask = ccMask;

% 7.1) DDQN 학습용 경험(train) 전송
%  state:   [flatten된 pfMetric, prevMask] (1×(numUE*numCC*2))
%  action:  scalar maskIdx (dec2bin → ccMask 변환에 사용된 값)
%  reward:  totalThroughput(Mbps) – λ·#Activated CCs
%  next_state: [다음 TTI 기준 pfMetric, ccMask]
%  done:    마지막 TTI 여부

% (1) 현 TTI 상태(state) 구성
pfVec      = pfMetric(:)';                       % PF metric flatten
prevMaskVec= prevMask(:)';                       % 이전 마스크 flatten
state      = [pfVec, prevMaskVec];

% (2) 페널티 보상 계산
NactiveSCell_perUE = sum(ccMask(:,2:end),2);           % [numUE×1]
totalSCells        = sum(NactiveSCell_perUE);          % 스칼라 (모든 UE 합)
R_base             = sum(lastThr(:))/1e6;              % 스칼라 total throughput (Mbps)
% 최종 보상: 스루풋 – γ·(전체 SCell 수)
reward             = R_base - simConfig.gamma * totalSCells;

% (3) 다음 상태(next_state) 구성
%    ttiHistory에 stateUpdate가 추가된 뒤 computePFMetric 호출 시뮬레이션 재현
prevHistory= [ttiHistory, stateUpdate];
nextPF     = computePFMetric(prevHistory);       % [numUE×numCC]
nextMask   = stateUpdate.CCMask;                 % logical matrix
nextPFVec  = nextPF(:)';                         
nextMaskVec= nextMask(:)';                       
next_state = [nextPFVec, nextMaskVec];

%% 8) 로그 반환
assert(numel(headers) == size(logData,2), 'Header/data mismatch: %d headers vs %d cols', numel(headers), size(logData,2));


logOut.headers = headers;
logOut.data    = logData;

%% 9) 경험 전송 (Train 메시지)
expStruct.type       = 'train';
expStruct.state      = state;
ccMaskVec            = ccMask(:);
expStruct.action     = maskIdx;
expStruct.reward     = reward;
expStruct.next_state = next_state;
expStruct.done       = (tti == simConfig.numTTI);

sendExperienceToPython(expStruct);

end

function [maskIdx, qValues] = sendInferenceToPython(pfVec, prevMaskVec)
    % Python REQ/REP 소켓을 이용해 DDQN 서버에 infer 요청 후 응답 받기

    % (1) import 문은 함수 시작 직후, persistent 앞에 위치해야 합니다
    import py.zmq.Context
    import py.zmq.REQ

    % (2) persistent 소켓 생성
    persistent sock
    if isempty(sock)
        ctx  = Context();
        sock = ctx.socket(REQ);
        sock.connect('tcp://127.0.0.1:5555');
    end

    % (3) JSON 메시지 구성
    msg = struct( ...
        'type',     'infer', ...
        'pfMetric', pfVec, ...
        'prevMask', prevMaskVec ...
    );
    jsonReq = jsonencode(msg);

    % (4) send → recv 반드시 쌍으로 실행
    sock.send_string(py.str(jsonReq));
    jsonResp = char( sock.recv_string() );

    % (5) JSON 응답 파싱
    resp    = jsondecode(jsonResp);
    maskIdx = resp.maskIdx;
    qValues = resp.qValues;
end