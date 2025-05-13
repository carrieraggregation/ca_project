function ccMask = decideCCActivation_pf(pfMetric, prevMask)
% decideCCActivation  PF 기반 CC 활성화 마스크 생성
% 입력:
%   pfMetric – [numUE × numCC] PF 메트릭 행렬
% 출력:
%   ccMask   – [numUE × numCC] logical, 활성화된 CC는 true

[numUE, numCC] = size(pfMetric);
% 초기화: 모두 비활성화
ccMask = false(numUE, numCC);
pctThr = 0.3;  % 상위 30% PF 지점

for ue = 1:numUE
    % CC1은 항상 활성화
    ccMask(ue,1) = true;
    if numCC > 1
        % 2) PF 상위 percentile 기준 CC2~n 활성
        met = pfMetric(ue,2:end);
        thr = quantile(met, 1-pctThr);
        sel = find(met >= thr) + 1;  % CC index 오프셋
        ccMask(ue, sel) = true;

        % 3) Hysteresis: 직전 TTI 활성 CC는 유지
        if nargin>1 && ~isempty(prevMask)
            ccMask(ue, :) = ccMask(ue,:) | prevMask(ue,:);
        end
    end
end
end