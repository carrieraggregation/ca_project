function rbAlloc = pfScheduler(pfMetric, ccMask, nRB, chunkSize)
% pfScheduler  Proportional Fair 기반 RB(Resource Block) 할당
% 입력:
%   pfMetric – [numUE × numCC] 크기의 PF 매트릭 행렬
%   ccMask   – [numUE × numCC] 논리형 CC 활성화 마스크
%   nRB      – CC당 총 Resource Block 수
% 출력:
%   rbAlloc – [numUE × numCC × nRB] 논리형 RB 할당 행렬

[numUE, numCC] = size(pfMetric);
% 결과 배열 초기화
rbAlloc = false(numUE, numCC, nRB);

% CC별 RB 할당
nChunks   = floor(nRB/chunkSize);
for cc = 1:numCC
    % 활성화된 UE 인덱스 추출
    activeUEs = find(ccMask(:, cc));
    if isempty(activeUEs)
        continue;
    end
    % 해당 CC의 PF 메트릭 추출
    metrics = pfMetric(activeUEs, cc);
    total   = sum(metrics);
    % PF 기반 확률 분포 계산
    if total > 0
        prob = metrics / total;
    else
        prob = ones(size(metrics)) / numel(metrics);
    end
    rbIndex = 1;
    % Step 1: CC1의 경우 모든 활성 UE에 최소 1 RB 보장
    if cc == 1
        for idx = 1:numel(activeUEs)
            if rbIndex > nRB, break; end
            ue = activeUEs(idx);
            rbAlloc(ue, cc, rbIndex) = true;
            rbIndex = rbIndex + 1;
        end
    end
    % → 대체: RB chunk 단위 룰렛휠 할당
    for ch = 1:nChunks
        sel = randsample(numel(activeUEs),1,true,prob);
        ue  = activeUEs(sel);
        % 이 chunkSize RB 모두 할당
        rbIdx = (ch-1)*chunkSize + (1:chunkSize);
        rbAlloc(ue,cc,rbIdx) = true;
    end
    % 남는 RB (nRB - chunkSize*nChunks)는 마지막 UE에게 clip
    rem = nRB - chunkSize*nChunks;
    if rem>0
        sel = randsample(numel(activeUEs),1,true,prob);
        ue  = activeUEs(sel);
        rbIdx = chunkSize*nChunks + (1:rem);
        rbAlloc(ue,cc,rbIdx) = true;
    end
end
end
