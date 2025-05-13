function headers = buildHeaders(numCC)
  headers = {'Seed','UE','X','Y','Z','Distance','AvgCQI'};
  for c = 1:numCC, headers{end+1} = sprintf('CC%d',c); end
  for c = 1:numCC, headers{end+1} = sprintf('CQI_CC%d',c); end
  for c = 1:numCC, headers{end+1} = sprintf('Thr_CC%d_Mbps',c); end
  headers{end+1} = 'TotalThr_Mbps';
end
