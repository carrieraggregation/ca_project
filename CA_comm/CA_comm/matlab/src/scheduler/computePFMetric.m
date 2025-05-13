function pfMetric = computePFMetric(ttiHistory)
% computePFMetric  PF 메트릭 계산
% 입력:
%   ttiHistory – struct array, 각 원소에 .lastThroughput, .avgThroughput 필드 포함
% 출력:
%   pfMetric   – numUE×numCC PF metric 행렬

    if isempty(ttiHistory)
        error('computePFMetric: ttiHistory is empty');
    end
    last = ttiHistory(end);
    if ~isfield(last,'lastThroughput') || ~isfield(last,'avgThroughput')
        error('computePFMetric: history missing fields');
    end
    inst = last.lastThroughput;
    avg  = last.avgThroughput;
    % 0 나눗셈 방지
    avg(avg==0) = 1;
    pfMetric = inst ./ avg;
end
