function [mimo_uplink] = PUSCH(type,NRB,f_up,Vc,UEAtNum,BSAtNum,DelayS,raseed)
%PUSCH 此处显示有关此函数的摘要
%   此处显示详细说明
%%
simParameters = [];             % Clear simParameters variable
simParameters.NFrames = 1;      % Number of 10ms frames
simParameters.SNRIn = [ 5]; % SNR range

%%
% The variable |displaySimulationInformation| controls the display of
% simulation information such as the HARQ process ID used for each
% subframe. In case of CRC error, the value of the index to the RV sequence
% is also displayed.

displaySimulationInformation = true;

%% UE and PUSCH Configuration
% Set the key parameters of the simulation. These include:
% 
% * The bandwidth in resource blocks (12 subcarriers per resource block)
% * Subcarrier spacing: 15, 30, 60, 120, 240 (kHz)
% * Cyclic prefix length: normal or extended
% * Cell ID
% * Number of transmit and receive antennas
% 
% A substructure containing the UL-SCH and PUSCH parameters is also
% specified. This includes:
% 
% * Target code rate
% * Allocated resource blocks (PRBSet)
% * Modulation scheme: 'pi/2-BPSK', 'QPSK', '16QAM', '64QAM', '256QAM'
% * Number of layers
% * Transform precoding (enable/disable)
% * PUSCH transmission scheme and MIMO precoding matrix indication (TPMI)
% * Number of antenna ports
% * PUSCH mapping type
% * DM-RS configuration parameters
% 
% Other simulation wide parameters are:
% 
% * Propagation channel model: 'TDL' or 'CDL'
%
% Note that if transform precoding is enabled, the number of layers should
% be set to 1.

% Bandwidth, numerology (SCS and CP type) and other general parameters
simParameters.NRB = NRB;                % Bandwidth in number of resource blocks (52RBs at 15kHz SCS for 10MHz BW)
simParameters.SubcarrierSpacing = 15;  % 15, 30, 60, 120, 240 (kHz)
simParameters.CyclicPrefix = 'Normal'; % 'Normal' or 'Extended'
simParameters.NCellID = 0;             % Cell identity
simParameters.NTxAnts = UEAtNum;             % Number of transmit antennas
simParameters.NRxAnts = BSAtNum;             % Number of receive antennas

% UL-SCH/PUSCH parameters
simParameters.PUSCH.TargetCodeRate = 193 / 1024;      % Code rate used to calculate transport block sizes
simParameters.PUSCH.PRBSet = (0:simParameters.NRB-1); % PUSCH PRB allocation
simParameters.PUSCH.SymbolSet = 0:13;            % PUSCH symbol allocation in each slot
simParameters.PUSCH.NohPRB = 0;                  % Additional RE overhead per PRB
simParameters.PUSCH.EnableHARQ = true;           % Enable/disable HARQ, if disabled, single transmission with RV=0, i.e. no retransmissions
simParameters.PUSCH.Modulation = '16QAM';         % 'pi/2-BPSK', 'QPSK', '16QAM', '64QAM', '256QAM'
simParameters.PUSCH.NLayers = 1;                 % Number of PUSCH layers
simParameters.PUSCH.RNTI = 1;                    % Radio Network Temporary Identifier
simParameters.PUSCH.TransformPrecoding = false;  % Enable/disable transform precoding
simParameters.PUSCH.TxScheme = 'nonCodebook';    % Transmission scheme ('nonCodebook','codebook')
simParameters.PUSCH.NAntennaPorts = 1;           % Number of antenna ports for codebook based precoding
simParameters.PUSCH.TPMI = 0;                    % Precoding matrix indicator for codebook based precoding
% PUSCH DM-RS configuration
simParameters.PUSCH.PUSCHMappingType = 'A';      % PUSCH mapping type ('A'(slot-wise),'B'(non slot-wise))
simParameters.PUSCH.DMRSTypeAPosition = 2;       % Mapping type A only. First DM-RS symbol position (2,3)
simParameters.PUSCH.DMRSLength = 1;              % Number of front-loaded DM-RS symbols (1(single symbol),2(double symbol))
simParameters.PUSCH.DMRSAdditionalPosition = 1;  % Additional DM-RS symbol positions (max range 0...3)
simParameters.PUSCH.DMRSConfigurationType = 1;   % DM-RS configuration type (1,2)
simParameters.PUSCH.NumCDMGroupsWithoutData = 2; % CDM groups without data
simParameters.PUSCH.NIDNSCID = 0;                % Scrambling identity (0...65535)
simParameters.PUSCH.NSCID = 0;                   % Scrambling initialization (0,1)
simParameters.PUSCH.NRSID = 0;                   % Scrambling ID for low-PAPR sequences (0...1007)
simParameters.PUSCH.GroupHopping = 'Disable';    % Hopping type ('Enable','Disable')

% Define the propagation channel type
simParameters.ChannelType = 'CDL'; % 'CDL' or 'TDL'

%%
% Create UE configuration structure |ue| and PUSCH configuration structure 
% |pusch|.

ue = simParameters;
pusch = simParameters.PUSCH;

%%
% For key simulation parameters, define local variables for convenience.

snrIn = simParameters.SNRIn;
nTxAnts = simParameters.NTxAnts;
nRxAnts = simParameters.NRxAnts;
channelType = simParameters.ChannelType;

%% 计算多普勒频移
v = Vc;                    % UT velocity in km/h                
c = physconst('lightspeed'); % speed of light in m/s
fd_up = ((v*1000/3600)/c) * f_up;     % UT max Doppler frequency in Hz

%% Propagation Channel Model Configuration
% Create the channel model object. Both CDL and TDL channel models are
% supported [ <#14 5> ].

if strcmpi(channelType,'CDL')
    
    channel = nrCDLChannel;
    channel.DelayProfile = type;
    [txsize,rxsize] = hArrayGeometry(nTxAnts,nRxAnts,'uplink');
    channel.TransmitAntennaArray.Size = txsize;
    channel.ReceiveAntennaArray.Size = rxsize;
    channel.Seed = raseed;
    channel.DelaySpread = DelayS; % in seconds
    channel.MaximumDopplerShift = fd_up; % in Hz
    channel.CarrierFrequency = f_up;
    
else
    
    channel = nrTDLChannel;
    channel.DelayProfile = 'TDL-A';
    channel.NumTransmitAntennas = nTxAnts;
    channel.NumReceiveAntennas = nRxAnts;
    
end



%%
% The sampling rate for the channel model is set using the value returned 
% from hOFDMInfo.

waveformInfo = hOFDMInfo(ue);
channel.SampleRate = waveformInfo.SamplingRate;

%%
% Get the maximum number of delayed samples by a channel multipath
% component. This is calculated from the channel path with the largest
% delay and the implementation delay of the channel filter. This is
% required later to flush the channel filter to obtain the received signal.

chInfo = info(channel);
maxChDelay = ceil(max(chInfo.PathDelays*channel.SampleRate));
maxChDelay = maxChDelay + chInfo.ChannelFilterDelay;

%% Processing Loop
% To determine the throughput at each SNR point, the PUSCH data is analyzed 
% per transmission instance using the following steps:
% 
% * _Update current HARQ process._ Check the CRC of the previous
% transmission for the given HARQ process. Determine whether a
% retransmission is required. If that is not the case generate new data.
% * _Generate resource grid._ Channel coding is performed by
% <docid:5g_ref#mw_sysobj_nrULSCH nrULSCH>. It operates on the input
% transport block provided. Internally, it keeps a copy of the transport
% block in case a retransmission is required. The coded bits are modulated
% by <docid:5g_ref#mw_function_nrPUSCH nrPUSCH>. Implementation-specific 
% MIMO precoding is applied to the resulting signal. Note that if 
% |TxScheme='codebook'|, codebook based MIMO precoding will already have
% been applied inside |nrPUSCH| and the implementation-specific MIMO 
% precoding is an additional stage of MIMO precoding. 
% * _Generate waveform._ The generated grid is then OFDM modulated.
% * _Model noisy channel._ The waveform is passed through a CDL or TDL
% fading channel. AWGN is added. The SNR for each layer is defined per RE
% and per receive antenna.
% * _Perform synchronization and OFDM demodulation._ Information returned
% by the channel is used for perfect synchronization. The synchronized
% signal is then OFDM demodulated.
% * _Perform perfect channel estimation._ Perfect channel estimation is
% used.
% * _Extract PUSCH and perform equalization._ The resource elements
% corresponding to the PUSCH allocation are extracted from the received
% OFDM resource grid and the channel estimate using
% <docid:5g_ref#mw_function_nrExtractResources nrExtractResources>. The
% received PUSCH resource elements are then MMSE equalized using
% <docid:5g_ref#mw_function_nrEqualizeMMSE nrEqualizeMMSE>.
% * _Decode the PUSCH._ The equalized PUSCH symbols, along with a noise
% estimate, are demodulated and descrambled by
% <docid:5g_ref#mw_function_nrPUSCHDecode nrPUSCHDecode> to obtain an
% estimate of the received codewords.
% * _Decode the Uplink Shared Channel (UL-SCH) and store the block CRC
% error for a HARQ process._ The vector of decoded soft bits is passed to
% <docid:5g_ref#mw_sysobj_nrULSCHDecoder nrULSCHDecoder> which decodes
% the codeword and returns the block CRC error used to determine the
% throughput of the system.

% Array to store the maximum throughput for all SNR points
maxThroughput = zeros(length(snrIn),1);
% Array to store the simulation throughput for all SNR points
simThroughput = zeros(length(snrIn),1);

% Set up Redundancy Version (RV) sequence, number of HARQ processes and
% the sequence in which the HARQ processes are used
if pusch.EnableHARQ
    % From PUSCH demodulation requirements in RAN WG4 meeting #88bis
    % (R4-1814062)
    rvSeq = [0 2 3 1];
else
    % HARQ disabled - single transmission with RV=0, no retransmissions
    rvSeq = 0;
end
% Specify the order in which we cycle through the HARQ processes
NHARQProcesses = 16;
harqSequence = 1:NHARQProcesses;

% Create UL-SCH encoder System object
encodeULSCH = nrULSCH;
encodeULSCH.MultipleHARQProcesses = true;
encodeULSCH.TargetCodeRate = pusch.TargetCodeRate;

% Create UL-SCH decoder System object
decodeULSCH = nrULSCHDecoder;
decodeULSCH.MultipleHARQProcesses = true;
decodeULSCH.TargetCodeRate = pusch.TargetCodeRate;

for snrIdx = 1:numel(snrIn)
    
    % Reset the random number generator and channel, so that each SNR point
    % will experience the same noise and channel realizations
    rng('default');
    reset(channel);
    
    % Initialize the state of all HARQ processes and reset the UL-SCH 
    % decoder
    harqProcesses = hNewHARQProcesses(NHARQProcesses,rvSeq,1);
    harqProcCntr = 0; % HARQ process counter
    reset(decodeULSCH);
    
    SNRdB = snrIn(snrIdx);
%     fprintf('\nSimulating %s-based transmission scheme (%dx%d) and SCS=%dkHz with %s channel at %gdB SNR for %d 10ms frame(s)\n',pusch.TxScheme,nTxAnts,nRxAnts,ue.SubcarrierSpacing,channelType,SNRdB,ue.NFrames);

    % Total number of OFDM symbols in the simulation period
    NSymbols = ue.NFrames * 10 * waveformInfo.SymbolsPerSubframe;
    
    % OFDM symbol number associated with start of each PUSCH transmission
    ue.NSymbol = 0;
    
    % Running counter of the number of PUSCH transmission instances
    % The simulation will use this counter as the slot number for each
    % PUSCH
    pusch.NSlot = 0;
    
    ii = 1;
    while ue.NSymbol < NSymbols

        % Calculate the transport block size for this slot
        [puschIndices,dmrsIndices,dmrsSymbols,puschIndicesInfo] = hPUSCHResources(ue,pusch);
        TBS = hPUSCHTBS(pusch,puschIndicesInfo.NREPerPRB - pusch.NohPRB);
        
        % Get HARQ process index for the current PUSCH from the HARQ index
        % table
        harqProcIdx = harqSequence(mod(harqProcCntr,length(harqSequence))+1);
        
        % Update current HARQ process information (this updates the RV
        % depending on CRC pass or fail in the previous transmission for
        % this HARQ process)
        harqProcesses(harqProcIdx) = hUpdateHARQProcess(harqProcesses(harqProcIdx),1);
        
        % HARQ: check CRC from previous transmission, i.e. is a
        % retransmission required?
        NDI = false;
        if harqProcesses(harqProcIdx).blkerr % errored
            if (harqProcesses(harqProcIdx).RVIdx==1) % end of rvSeq
                resetSoftBuffer(decodeULSCH,harqProcIdx-1);
                NDI = true;
            end
        else % no error
            NDI = true;
        end
        if NDI 
            trBlk = randi([0 1],TBS,1);
            setTransportBlock(encodeULSCH,trBlk,harqProcIdx-1);
        end
        
        % UL-SCH encoding
        codedTrBlock = encodeULSCH(pusch.Modulation,pusch.NLayers,puschIndicesInfo.G,harqProcesses(harqProcIdx).RV,harqProcIdx-1);
        
        % PUSCH modulation, including codebook based MIMO precoding if 
        % TxScheme = 'codebook'
        MRB = numel(pusch.PRBSet);
        puschSymbols = nrPUSCH(codedTrBlock,pusch.Modulation,pusch.NLayers,ue.NCellID,pusch.RNTI,pusch.TransformPrecoding,MRB,pusch.TxScheme,pusch.NAntennaPorts,pusch.TPMI);
        
        % Create resource grid associated with PUSCH transmission period
        puschGrid = zeros(waveformInfo.NSubcarriers,waveformInfo.SymbolsPerSlot,nTxAnts);
        
        % Implementation-specific PUSCH MIMO precoding and mapping. This 
        % MIMO precoding step is in addition to any codebook based 
        % MIMO precoding done during PUSCH modulation above
        if (strcmpi(pusch.TxScheme,'codebook'))
            % codebook based MIMO precoding, F precodes between PUSCH
            % transmit antenna ports and transmit antennas
            F = eye(pusch.NAntennaPorts,nTxAnts);
        else
            % non-codebook based MIMO precoding, F precodes between PUSCH 
            % layers and transmit antennas
            F = eye(pusch.NLayers,nTxAnts);
        end
        [~,puschAntIndices] = nrExtractResources(puschIndices,puschGrid);
        puschGrid(puschAntIndices) = puschSymbols * F;
        
        % Implementation-specific PUSCH DM-RS MIMO precoding and mapping.
        % The DM-RS creation in hPUSCHResources above includes codebook
        % based MIMO precoding if applicable
        for p = 1:size(dmrsSymbols,2)
            [~,dmrsAntIndices] = nrExtractResources(dmrsIndices(:,p),puschGrid);
            puschGrid(dmrsAntIndices) = puschGrid(dmrsAntIndices) + dmrsSymbols(:,p) * F(p,:);
        end
        
        % OFDM modulation
        txWaveform = hOFDMModulate(ue,puschGrid);

        % Pass data through channel model. Append zeros at the end of the
        % transmitted waveform to flush channel content. These zeros take
        % into account any delay introduced in the channel. This is a mix
        % of multipath delay and implementation delay. This value may 
        % change depending on the sampling rate, delay profile and delay
        % spread
        txWaveform = [txWaveform; zeros(maxChDelay,size(txWaveform,2))]; %#ok<AGROW>
        [rxWaveform,pathGains,sampleTimes] = channel(txWaveform);
        
        % Add AWGN to the received time domain waveform 
        % Normalize noise power by the IFFT size used in OFDM modulation,
        % as the OFDM modulator applies this normalization to the
        % transmitted waveform. Also normalize by the number of receive
        % antennas, as the default behaviour of the channel model is to
        % apply this normalization to the received waveform
        SNR = 10^(SNRdB/20);
        N0 = 1/(sqrt(2.0*nRxAnts*double(waveformInfo.Nfft))*SNR);
        noise = N0*complex(randn(size(rxWaveform)),randn(size(rxWaveform)));
        rxWaveform = rxWaveform + noise;

        % Perfect synchronization. Use information provided by the channel
        % to find the strongest multipath component
        pathFilters = getPathFilters(channel);
        [offset,mag] = nrPerfectTimingEstimate(pathGains,pathFilters);
        rxWaveform = rxWaveform(1+offset:end,:);

        % Perform OFDM demodulation on the received data to recreate the
        % resource grid
        rxGrid = hOFDMDemodulate(ue,rxWaveform);
        
        % Perfect channel estimation, use the value of the path gains
        % provided by the channel
        estChannelGrid = nrPerfectChannelEstimate(pathGains,pathFilters,ue.NRB,ue.SubcarrierSpacing,pusch.NSlot,offset,sampleTimes,ue.CyclicPrefix);
        HUL(ii,:,:,:,:) = estChannelGrid;ii = ii+1;
        
        % Get perfect noise estimate (from the noise realization)
        noiseGrid = hOFDMDemodulate(ue,noise(1+offset:end,:));
        noiseEst = var(noiseGrid(:));
        
        % Apply MIMO precoding to estChannelGrid
        % Linearize 4D matrix and reshape after multiplication
        K = size(estChannelGrid,1);
        estChannelGrid = reshape(estChannelGrid,K*waveformInfo.SymbolsPerSlot*nRxAnts,nTxAnts);
        estChannelGrid = estChannelGrid * F.';
        if (strcmpi(pusch.TxScheme,'codebook'))
            W = nrPUSCHCodebook(pusch.NLayers,pusch.NAntennaPorts,pusch.TPMI,pusch.TransformPrecoding);
            estChannelGrid = estChannelGrid * W.';
        end
        estChannelGrid = reshape(estChannelGrid,K,waveformInfo.SymbolsPerSlot,nRxAnts,[]);
        
        % Get PUSCH resource elements from the received grid
        [puschRx,puschHest] = nrExtractResources(puschIndices,rxGrid,estChannelGrid);
        
        % Equalization
        [puschEq,csi] = nrEqualizeMMSE(puschRx,puschHest,noiseEst);
        
        % Decode PUSCH physical channel
        [ulschLLRs,rxSymbols] = nrPUSCHDecode(puschEq,pusch.Modulation,ue.NCellID,pusch.RNTI,noiseEst,pusch.TransformPrecoding,MRB);
        
        % Apply channel state information (CSI) produced by the equalizer,
        % including the effect of transform precoding if enabled
        if (pusch.TransformPrecoding)
            MSC = MRB * 12;
            csi = nrTransformDeprecode(csi,MRB) / sqrt(MSC);
            csi = repmat(csi((1:MSC:end).'),1,MSC).';
            csi = reshape(csi,size(rxSymbols));
        end
        csi = nrLayerDemap(csi);
        Qm = length(ulschLLRs) / length(rxSymbols);
        csi = reshape(repmat(csi{1}.',Qm,1),[],1);
        ulschLLRs = ulschLLRs .* csi;
        
        % Decode the UL-SCH transport channel
        decodeULSCH.TransportBlockLength = TBS;
        [decbits,harqProcesses(harqProcIdx).blkerr] = decodeULSCH(ulschLLRs,pusch.Modulation,pusch.NLayers,harqProcesses(harqProcIdx).RV,harqProcIdx-1);
        
        % Store values to calculate throughput
        simThroughput(snrIdx) = simThroughput(snrIdx) + (~harqProcesses(harqProcIdx).blkerr * TBS);
        maxThroughput(snrIdx) = maxThroughput(snrIdx) + TBS;
        
        % Display transport block error information
        if (displaySimulationInformation)
%             fprintf('\n(%3.2f%%) HARQ Proc %d: ',100*(ue.NSymbol+size(puschGrid,2))/NSymbols,harqProcIdx);
%             estrings = {'passed','failed'};
            rvi = harqProcesses(harqProcIdx).RVIdx;
            if rvi == 1
                ts = sprintf('Initial transmission (RV=%d)',rvSeq(rvi));
            else
                ts = sprintf('Retransmission #%d (RV=%d)',rvi-1,rvSeq(rvi));
            end
%             fprintf('%s %s. ',ts,estrings{1+harqProcesses(harqProcIdx).blkerr});
        end
        
        % Update starting symbol number of next PUSCH transmission
        ue.NSymbol = ue.NSymbol + size(puschGrid,2);
        
        % Update count of overall number of PUSCH transmissions
        pusch.NSlot = pusch.NSlot + 1;
        
        % Update HARQ process counter
        harqProcCntr = harqProcCntr + 1;

    end
    
    % Display the results dynamically in the command window
    if (displaySimulationInformation)
%         fprintf('\n');
    end
%     fprintf([['\nThroughput(Mbps) for ' num2str(ue.NFrames) ' frame(s) '],'= %.4f\n'], 1e-6*simThroughput(snrIdx)/(ue.NFrames*10e-3));
%     fprintf(['Throughput(%%) for ' num2str(ue.NFrames) ' frame(s) = %.4f\n'],simThroughput(snrIdx)*100/maxThroughput(snrIdx));

end

%% Results
% Display the measured throughput. This is calculated as the percentage of
% the maximum possible throughput of the link given the available resources
% for data transmission.

% figure;
% plot(snrIn,simThroughput*100./maxThroughput,'o-.')
% xlabel('SNR (dB)'); ylabel('Throughput (%)'); grid on;
% if (pusch.TransformPrecoding)
%     ofdmType = 'DFT-s-OFDM';
% else
%     ofdmType = 'CP-OFDM';
% end
% title(sprintf('%s / NRB=%d / SCS=%dkHz / %s %d/1024 / %dx%d', ...
%     ofdmType,ue.NRB,ue.SubcarrierSpacing,pusch.Modulation, ...
%     round(pusch.TargetCodeRate*1024),nTxAnts,nRxAnts));

simResults.simParameters = simParameters;
simResults.simThroughput = simThroughput;
simResults.maxThroughput = maxThroughput;

%%
% The figure below shows throughput results obtained simulating 10000
% subframes (|NFrames = 1000|, |SNRIn = -16:2:6|).
%
% <<../longRunPUSCHThroughput.png>>
%

%% Appendix
% This example uses the following helper functions:
%
% * <matlab:edit('hArrayGeometry.m') hArrayGeometry.m>
% * <matlab:edit('hNewHARQProcesses.m') hNewHARQProcesses.m>
% * <matlab:edit('hOFDMDemodulate.m') hOFDMDemodulate.m>
% * <matlab:edit('hOFDMInfo.m') hOFDMInfo.m>
% * <matlab:edit('hOFDMModulate.m') hOFDMModulate.m>
% * <matlab:edit('hPUSCHTBS.m') hPUSCHTBS.m>
% * <matlab:edit('hPUSCHResources.m') hPUSCHResources.m>
% * <matlab:edit('hUpdateHARQProcess.m') hUpdateHARQProcess.m>

%% Selected Bibliography
% # 3GPP TS 38.211. "NR; Physical channels and modulation (Release 15)."
% 3rd Generation Partnership Project; Technical Specification Group Radio
% Access Network.
% # 3GPP TS 38.212. "NR; Multiplexing and channel coding (Release 15)." 3rd
% Generation Partnership Project; Technical Specification Group Radio
% Access Network.
% # 3GPP TS 38.213. "NR; Physical layer procedures for control (Release
% 15)." 3rd Generation Partnership Project; Technical Specification Group
% Radio Access Network.
% # 3GPP TS 38.214. "NR; Physical layer procedures for data (Release 15)."
% 3rd Generation Partnership Project; Technical Specification Group Radio
% Access Network.
% # 3GPP TR 38.901. "Study on channel model for frequencies from 0.5 to 100
% GHz (Release 15)." 3rd Generation Partnership Project; Technical
% Specification Group Radio Access Network.

mimo_uplink = HUL(5,:,:,:,:);

end

