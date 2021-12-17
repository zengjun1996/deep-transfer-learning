%hPUSCHTBS 5G PUSCH transport block size determination
%   TBS = hPUSCHTBS(CHS,NDRE) returns the transport block sizes associated
%   with a PUSCH transmission, as defined in TS 38.214 6.1.4.2.
%
%   The PUSCH specific configuration input, CHS, must be a structure
%   including the fields:
%   TargetCodeRate      - Code rate used to calculate transport block sizes
%   PRBSet              - PRBs allocated to the PUSCH (0-based indices)
%   NLayers             - Total number of layers
%   Modulation          - Modulation scheme ('pi/2-BPSK','QPSK','16QAM','64QAM','256QAM')
%
%   The second input, NDRE, is the number of RE allocated per PRB to
%   the PUSCH (N'RE) in the slot, accounting for DM-RS, CDM groups and any
%   additional overhead (Xoh-PUSCH).
% 
%   Example:
%    
%   pusch = struct();
%   pusch.TargetCodeRate = 0.5;
%   pusch.PRBSet = [0:99];
%   pusch.NLayers = 2;
%   pusch.Modulation = '16QAM';
%   nred = 12*14;
% 
%   tbs = hPUSCHTBS(pusch,nred)
% 
%   See also nrULSCH, nrULSCHDecoder, hPUSCHResources.

%   Copyright 2019 The MathWorks, Inc.

function tbs = hPUSCHTBS(pusch,nred)
    
    % The effective calculation for the PUSCH is the same as for the PDSCH
    tbs = hPDSCHTBS(pusch,nred);
    
end
