%% File: simulateLinkPHY.m
% simulateLinkPHY  OFDM → TDL 채널 → AWGN → MMSE → BER/Throughput
% 입력:
%   snr_dB       – faded SNR (scalar)
%   modulation   – 'QPSK','16QAM','64QAM','256QAM'
%   simConfig    – struct with fields nSC, nSymbols, SubcarrierSpacing
%   allocatedRBs – RB 개수 (scalar)
%   chanObj      – nrTDLChannel 객체 for this UE-CC
% 출력:
%   ber          – BER (scalar)
%   throughput_bps – 처리 비트수 per TTI (bps)
function [ber, throughput_bps] = simulateLinkPHY(snr_dB, modulation, simConfig, allocatedRBs, chanObj)
    % 1) OFDM 파라미터
    M_map      = containers.Map({'QPSK','16QAM','64QAM','256QAM'}, [4,16,64,256]);
    M          = M_map(char(modulation));
    Qm         = log2(M);
    FFT_Size   = allocatedRBs * simConfig.nSC;
    GI_Size    = FFT_Size / 4;
    numLayers  = 2;                          % fixed 2×2 MIMO
    noisePower = 10^(-snr_dB/10);

    % 2) 비트 생성 및 QAM 변조
    txBits = randi([0 1], numLayers, FFT_Size * Qm);
    modData = zeros(numLayers, FFT_Size);
    for L = 1:numLayers
        bits    = reshape(txBits(L,:), Qm, []).';
        symbols = bi2de(bits, 'left-msb');
        modData(L,:) = qammod(symbols, M, 'gray', 'InputType','integer','UnitAveragePower',true);
    end

    % 3) OFDM IFFT + CP
    ifftData = ifft(modData, FFT_Size, 2) * sqrt(FFT_Size);
    txSignal = [ifftData(:, end-GI_Size+1:end), ifftData];  % [numLayers×(GI+FFT)]

    % 4) TDL 채널 통과 + AWGN
    rxMat = chanObj(txSignal.');                 % [time×numLayers]
    rxSig = rxMat.' + sqrt(noisePower/2)*(randn(size(rxMat.')) + 1j*randn(size(rxMat.')));

    % 5) Remove CP and FFT
    rxFFT = zeros(numLayers, FFT_Size);
    for L = 1:numLayers
        seg = rxSig(L, GI_Size+1:GI_Size+FFT_Size);
        rxFFT(L,:) = fft(seg, FFT_Size) / sqrt(FFT_Size);
    end

    % 6) MMSE Equalization
    H_fft = getChannelResponse(chanObj, FFT_Size);
    rxStreams = cell(1, numLayers);
    for k = 1:FFT_Size
        Hk = squeeze(H_fft(:,:,k));              % [numLayers×numLayers]
        yk = rxFFT(:,k);                        % [numLayers×1]
        W  = (Hk' * Hk + noisePower*eye(numLayers)) \ Hk';
        xHat = W * yk;
        for L = 1:numLayers
            demod = qamdemod(xHat(L), M, 'gray', 'OutputType','integer','UnitAveragePower',true);
            bits  = de2bi(demod, Qm, 'left-msb');
            rxStreams{L} = [rxStreams{L}, reshape(bits.',1,[])];
        end
    end

    % 7) BER 및 Throughput
    txAll = txBits(1,:);
    rxAll = rxStreams{1};
    errors = sum(txAll ~= rxAll);
    ber    = errors / numel(txAll);
    ttiLen_s = 1e-3;  % 1 ms
    throughput_bps = (numel(txAll) - errors) * simConfig.nSymbols / ttiLen_s;
end