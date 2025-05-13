%% File: getChannelResponse.m
% getChannelResponse  TDL 채널 객체로부터 서브캐리어별 주파수 응답 생성
% 입력:
%   chan      – nrTDLChannel 객체
%   FFT_Size  – OFDM IFFT/FFT 크기
% 출력:
%   H_fft     – [NumRx×NumTx×FFT_Size] complex frequency response
function H_fft = getChannelResponse(chan, FFT_Size)
    % 1) 경로 필터 계수 추출
    pfilt = getPathFilters(chan);
    dims  = size(pfilt);
    Nh    = dims(1);
    Nt    = chan.NumTransmitAntennas;
    Nr    = chan.NumReceiveAntennas;

    % 2) 시간응답 벡터 htime 추출
    switch numel(dims)
        case 2
            % [Nh×M] → 첫 열 사용
            htime = pfilt(:,1);
        case 3
            % [Nh×Nt×Nr] or [Nh×Nr×Nt]
            if dims(2) == Nt && dims(3) == Nr
                % [Nh×Nt×Nr] → sum over transmit antennas
                tmp   = squeeze(sum(pfilt,2));    % [Nh×Nr]
                htime = mean(tmp,2);
            elseif dims(2) == Nr && dims(3) == Nt
                % [Nh×Nr×Nt] → sum over transmit antennas
                tmp   = squeeze(sum(pfilt,3));    % [Nh×Nr]
                htime = mean(tmp,2);
            else
                error('getChannelResponse: unexpected 3D dims %s', mat2str(dims));
            end
        case 4
            % [Nh×1×Nr×Nt] typical for getPathFilters
            tmp   = squeeze(pfilt(:,1,:,:));      % [Nh×Nr×Nt]
            sumTx = squeeze(sum(tmp,3));          % [Nh×Nr]
            htime = mean(sumTx,2);
        otherwise
            error('getChannelResponse: unsupported pfilt dims %s', mat2str(dims));
    end

    % 3) 주파수 응답 생성
    fvec  = fft(htime, FFT_Size);            % [FFT_Size×1]
    H_fft = repmat(reshape(fvec,1,1,[]), [Nr, Nt, 1]);
end