%hPUSCHResources 5G NR PUSCH and DM-RS resource element indices and DM-RS values
%   [IND,DMRSIND,DMRS,INFO] = hPUSCHResources(UE,CHS) returns the 
%   resource element (RE) indices for a 5G NR PUSCH transmission and 
%   associated PUSCH DM-RS, given the time (OFDM symbols) and frequency 
%   (PRBs) allocation of the PUSCH and the DM-RS configuration. The 1-based
%   linear PUSCH indices are returned in matrix IND. They are defined
%   relative to a three-dimensional RE grid representing a 14/12-symbol
%   slot for NRB resource blocks (in the PUSCH numerology) across the 
%   DM-RS ports/layers of the PUSCH. Each column of IND represents the 
%   grid locations for a separate port or layer (the third dimension of 
%   the RE grid). The DM-RS RE indices have the same format and are 
%   returned in matrix DMRSIND. The complex values of associated DM-RS
%   sequence are also returned in matrix DMRS. Additional information 
%   the DM-RS and resourcing is returned in the structure INFO.
%  
%   The UE-wide settings input, UE, must be a structure including 
%   the fields:
%   NRB               - Number of resource blocks for the carrier
%                       or bandwidth part (in the PUSCH numerology)
%   CyclicPrefix      - Optional. Cyclic prefix length 
%                       ('Normal'(default),'Extended')
%   SubcarrierSpacing - Optional. Subcarrier Spacing (kHz)
%                       (15(default),30,60,120,240)
%
%   The PUSCH specific input, CHS, must be a structure including the
%   fields:
%   NSlot                   - Optional. Slot number of PUSCH transmission
%                             (default 0, can be the absolute slot number)
%   PRBSet                  - PRBs allocated to the PUSCH (0-based indices)
%   PRBRefPoint             - Optional. PRB reference offset that PRBSet 
%                             is relative to (0-based) (default 0)
%   SymbolSet               - OFDM symbols allocated to PUSCH within slot
%                             (including DM-RS symbols, 0-based indices,
%                             max range 0...13)
%   PortSet                 - DM-RS antenna ports used by PUSCH
%                             (0-based indices, max range 0...11, mapping
%                             to ports p=0...11 respectively)
%   NLayers                 - Optional. Number of transmission layers 
%                             (only used if PortSet was not defined and 
%                             then it selects ports=0...NLayers-1)
%   Modulation              - Modulation type of codeword
%                             ('pi/2-BPSK','BPSK','QPSK','16QAM','64QAM','256QAM')
%   Reserved                - Reserved PRB patterns (structure array, see below)
%   TransformPrecoding      - Optional. Transform precoding for SC-FDMA
%                             (0(default),1)
%     Required when transform precoding is enabled:
%       NRSID               - Scrambling ID for low-PAPR sequences (0..1007)
%       GroupHopping        - Hopping type ('Enable','Disable') 
%   TxScheme                - Optional. Transmission scheme
%                             ('NonCodebook'(default),'Codebook')
%     Required when transmission scheme is codebook based:
%       NAntennaPorts       - Number of antenna ports after codebook precoding
%       TPMI                - Transmitted precoding matrix indicator 
%                             (max range 0...27)
%   IntraSlotFreqHopping    - Optional. Intra-slot frequency hopping
%                             ('Disabled'(default),'Enabled')
%     Required when intra-slot frequency hopping is enabled:
%       RBOffset            - Second hop RB position index (0-based)
%   DMRSConfigurationType   - DM-RS configuration type (1,2)
%                             (Transform precoded case always uses type 1)
%   NumCDMGroupsWithoutData - Optional. Number of CDM groups without data
%                             (0(default),1,2,3)
%                             (Transform precoded case always uses 2)
%   NIDNSCID                - DM-RS scrambling identity (0...65535)
%                             (only used when transform precoding is disabled)
%   NSCID                   - DM-RS scrambling initialization (0,1)
%                             (only used when transform precoding is disabled)
%                            
%   The DM-RS OFDM symbol locations can either be defined explicitly 
%   using the single parameter:
%   DMRSSymbolSet           - OFDM symbols containing the DM-RS within the 
%                             PUSCH allocation in slot 
%                             (0-based indices, max range 0...13)
%   or, defined implicitly via the following group of DM-RS parameters:
%   PUSCHMappingType        - PUSCH mapping type
%                             ('A'(slot-wise),'B'(non slot-wise))
%   DMRSTypeAPosition       - Mapping type A only. First DM-RS symbol position
%                             (2,3)
%   DMRSLength              - Number of front-loaded DM-RS symbols
%                             (1(single symbol),2(double symbol))
%   DMRSAdditionalPosition  - Additional DM-RS symbol positions
%                             (max range 0...3)
% 
%   Periodically recurring patterns of reserved PRB can defined using the 
%   'Reserved' parameter. These PRB will be excluded from the generated 
%   indices and the UL-SCH/PUSCH processing should rate-match around them. 
%   This parameter takes the format of a structure array where each element
%   defines a separate pattern. 
%   Each element, i, of the array should contain the following fields:
%   Reserved(i).PRB     - Reserved PRB (0-based indices, defined as a 
%                         vector or cell array)
%   Reserved(i).Symbols - OFDM symbols associated with reserved PRB 
%                         (0-based indices, spanning one or more slots)
%   Reserved(i).Period  - Total number of slots in the pattern period
% 
%   The reserved PRB indices can be specified as a vector or a cell array.
%   If a vector then the same PRBs are excluded in each OFDM symbol in the 
%   pattern. If the PRB indices are defined as a cell array then each cell
%   specifies the excluded PRBs for the associated OFDM symbol in the 
%   pattern. In the latter case, the length of the PRB cell array should
%   match the length of the 'Symbols' field, i.e. an individual set of PRBs
%   is defined for each reserved symbol. The symbols that form the 
%   time-domain locations of the reserved pattern can be greater than 13
%   and therefore cover multiple slots. The overall pattern will repeat
%   itself every 'Period' slots. If this field is empty then the pattern 
%   will not cyclically repeat itself.
% 
%   In terms of frequency domain DM-RS density, there are two different RRC
%   signaled configuration types ('dmrs-Type'). Configuration type 1 
%   defines 6 subcarriers per PRB per antenna port, comprising alternating
%   subcarriers. Configuration type 2 defines 4 subcarriers per PRB per
%   antenna ports, consisting of 2 groups of 2 neighbouring subcarriers.
%   Different shifts are applied to the sets of subcarriers used, dependent
%   on the associated antenna port or CDM group. For type 1, there are 
%   2 possible CDM groups/shifts across up to 8 possible antenna ports
%   (p=0...7), and, for type 2, there are 3 possible CDM groups/shifts
%   across 12 ports (p=0...11). See TS 38.211 section 6.4.1.1 for the full
%   configuration details.
%
%   In terms of the time-domain DM-RS symbol positions, the PUSCH mapping
%   type ('PUSCHMappingType') can be either slot-wise (type A) or non
%   slot-wise (type B). When a UE is scheduled to receive PUSCH by a DCI,
%   this mapping type is signaled by the time-domain resource field in the
%   grant. The field acts as an index into an RRC configured table where
%   each row in the table specifies a combination of mapping type, slot
%   offset, K0, the symbol start and length indicator, SLIV. The mapping
%   type specifies the relative locations of the associated DM-RS. For
%   slot-wise mapping type A, the first DM-RS symbol is signaled by a field
%   in the MIB to be either 2 or 3 ('dmrs-TypeA-Position'). For the non
%   slot-wise mapping type B, the first DM-RS symbol is always the first
%   symbol of the PUSCH time allocation.
% 
%   The maximum number of DM-RS OFDM symbols used by a UE is configured by
%   RRC signaling ('dmrs-TypeA-Position' and 'maxLength'). The DM-RS
%   can be a set of single symbols, distributed roughly uniformly across 
%   the allocated PUSCH symbols, or 1 or 2 pairs of neighbouring or 'double 
%   symbol' DM-RS. The 'maxLength' RRC parameter (1 or 2 respectively)
%   configures whether only single symbol DM-RS or either single or double
%   symbol DM-RS can be used. In the latter case, the actual selection is 
%   signaled in the DCI format 0_1 message. The 'dmrs-TypeA-Position'
%   higher-layer parameter defines the number of single or double symbol
%   DM-RS that are transmitted. The valid combinations of these two 
%   parameters is given by TS 38.211 tables 6.4.1.1.3-3, 6.4.1.1.3-4 and 
%   6.4.1.1.3-6. In this function, the value of the 'DMRSLength' input
%   parameter directly controls whether either single or double symbols 
%   are used.
% 
%   INFO is the output structure containing the fields:
%   G             - Bit capacity of the PUSCH. This is should be the
%                   length of codeword to be output from the UL-SCH 
%                   transport channel
%   Gd            - Number of resource elements per layer/port, equal to
%                   the number of rows in the PUSCH indices
%   NREPerPRB     - Number of RE per PRB allocated to PUSCH (not accounting
%                   for any reserved resources)
%   DMRSSymbolSet - The symbol numbers in a slot containing DM-RS (0-based)
%   CDMGroups     - CDM groups associated with the DM-RS antenna ports
%
%   Example:
%   % Display the locations of the PUSCH and PUSCH DM-RS resource elements 
%   % in a slot. 
%     
%   % Set the number of uplink carrier or BWP resource blocks and the 
%   % numerology (subcarrier spacing and cyclic prefix)
%   ue = struct('NRB',50);
%   ue.SubcarrierSpacing = 15;
%   ue.CyclicPrefix = 'Normal';
%   
%   % Get the number of OFDM symbols in a slot
%   symbperslot = sum(strcmpi(ue.CyclicPrefix,["Normal","Extended"]) .* [14 12]);
%   
%   % Specify the basic PUSCH allocation properties to be full band, full
%   % slot using 2 layers/ports (no transform precoding or frequency hopping)
%   pusch = struct();
%   pusch.NSlot = 0;                   % Slot number
%   pusch.PRBSet = 0:ue.NRB-1;         % Full band PRBs allocated to the PUSCH
%   pusch.SymbolSet = 0:symbperslot-1; % Full slot allocation top the PUSCH
%   pusch.PortSet = [0,2];             % Use DM-RS ports p=0 and p=2
%   pusch.Modulation = 'QPSK';         % Modulation scheme
% 
%   % Configure the PUSCH DM-RS for config type 1 and slot-wise, type A
%   % mapping. Use double symbols for front-loaded DM-RS and configure an
%   % additional symbol pair towards the end of the slot
%   pusch.DMRSConfigurationType = 1;  % DM-RS configuration type (1,2)
%   pusch.NIDNSCID = 1;               % DM-RS scrambling identity (0...65535)
%   pusch.NSCID = 0;                  % DM-RS scrambling initialization (0,1)
%   pusch.PUSCHMappingType = 'A';     % Slot-wise PUSCH mapping type
%   pusch.DMRSTypeAPosition = 2;      % First DM-RS symbol position for type A
%   pusch.DMRSLength = 2;             % Specify double front-loaded DM-RS
%   pusch.DMRSAdditionalPosition = 1; % Specify an additional DM-RS pair
%   pusch.NumCDMGroupsWithoutData = 3;% CDM groups without data
% 
%   % Display PUSCH and DM-RS RE locations of the first port of the grid
%   slotgrid = zeros(12*ue.NRB,symbperslot,length(pusch.PortSet));
%   [ind,dmrsind,dmrs,info] = hPUSCHResources(ue,pusch);
%   slotgrid(ind) = 20;                 % Use light blue for PUSCH RE 
%   slotgrid(dmrsind) = 50*abs(dmrs);   % Use yellow for DM-RS RE
%   figure;
%   image(slotgrid(:,:,1));
%   title('PUSCH and DM-RS resource elements');
%   axis('xy'), xlabel('Symbols'), ylabel('Subcarriers');

%   Copyright 2019 The MathWorks, Inc.

function [puschIndices,dmrsIndices,dmrsSymbols,puschInfo] = hPUSCHResources(ue,pusch)
 
    % Argument check 
    narginchk(2,2);
 
    % Configure general callback information
    ftable.MappingType = 'PUSCHMappingType';
    ftable.dmrsSymbolsTable = @lookupPUSCHDMRSymbols;      % PUSCH DM-RS symbols numbers for transmission in a slot
    
    % Fix the constant settings when transform precoding is enabled
    if isfield(pusch,'TransformPrecoding') && pusch.TransformPrecoding
        pusch.DMRSConfigurationType = 1;            % Only type 1 can be used with transform precoding
        pusch.NumCDMGroupsWithoutData = 2;          % All type 1 CDM groups are reserved in this case and there is no data in the DM-RS SC-FDMA symbols
        ftable.getDMRSSequence = @lowPAPRSequence;  % Use the low PAPR sequences for the base DM-RS sequence
    end

    % If intra slot frequency hopping is enabled then create the two
    % different PRB sets for the first and second hop and place in a cell
    % array aligned with the PUSCH allocation length
    if isfield(pusch,'IntraSlotFreqHopping') && strcmpi(pusch.IntraSlotFreqHopping,'enabled') && ~iscell(pusch.PRBSet)
        nsymbols = length(pusch.SymbolSet);
        fhoplen = fix(nsymbols/2);  % First hop duration
        prbcell = cell(1,nsymbols);
        prbcell(1:fhoplen) = {pusch.PRBSet};
        [lprb,uprb] = bounds(pusch.PRBSet);
        if uprb-lprb+pusch.RBOffset >= ue.NRB
            error('Combination of the intra-slot frequency hopping offset RBOffset (%d) and the PRBSet exceeds the number of resource blocks NRB (%d)',pusch.RBOffset,ue.NRB);
        end    
        prbcell(fhoplen+1:end) = {pusch.PRBSet-lprb+pusch.RBOffset};
        pusch.PRBSet = prbcell;
    end    
    
    % Get the resource element indices and DM-RS symbols
    [puschIndices,dmrsIndices,dmrsSymbols,puschInfo,genInfo] = hSharedChannelResources(ftable,ue,pusch);
    
    % Apply codebook precoding matrix to the DM-RS symbols 
    % In the non-codebook case the precoding matrix is just the identity matrix
    if isfield(pusch,'TxScheme') && strcmpi(pusch.TxScheme,'codebook')

        % Use the CDM group number to label the DM-RS and group them into 
        % different sets of rows. Within a PRB, we know that the lower CDM
        % groups have lower delta shifts
        [groups,~,cdmgroupsidx] = unique(puschInfo.CDMGroups);
        cdmgroupsidx = reshape(cdmgroupsidx,1,[]);       % Need a row for implicit expansion below
        ngroups = length(groups);
        
        % DM-RS symbols
        pdmrsSymbols = zeros([ngroups 1].*size(dmrsSymbols));   % Expand by number of unique groups
        indices = (cdmgroupsidx-1) + reshape(1:ngroups:ngroups*numel(dmrsSymbols),[],size(dmrsSymbols,2));  
        pdmrsSymbols(indices) = dmrsSymbols;
        
        % DM-RS indices
        % Merge the indices used by the CDM groups
        % Each row is only associated with a single CDM
        nreperslot = 12*genInfo.SymbolsPerSlot*ue.NRB;
        normed = dmrsIndices-((0:size(dmrsIndices,2)-1)*nreperslot);
        pdmrsIndices(mod(indices(:)-1,size(pdmrsSymbols,1))+1) = normed;
       
        % Get the precoding matrix from the codebook
        if size(pdmrsSymbols,2) > pusch.NAntennaPorts 
            error('When using the codebook based PUSCH transmission scheme, the number of antenna ports (%d) cannot be less than the number of DM-RS ports (%d).',...
                        pusch.NAntennaPorts,size(pdmrsSymbols,2));
        end
        W = nrPUSCHCodebook(size(pdmrsSymbols,2),pusch.NAntennaPorts,pusch.TPMI,pusch.TransformPrecoding);
        
        % Apply codebook matrix to symbols and match up the indices
        % across the projected antenna ports
        dmrsSymbols = pdmrsSymbols * W;
        planeoffsets = nreperslot*(0:size(dmrsSymbols,2)-1);
        dmrsIndices = planeoffsets + repmat(pdmrsIndices',[1 size(dmrsSymbols,2)]);
        
        % Project the PUSCH indices across the number of output antenna ports
        puschIndices = planeoffsets + repmat(puschIndices(:,1),[1 size(dmrsSymbols,2)]);
        
    end
  
end

% Create a low-PAPR based DM-RS sequence for the transform precoding/SC-FDMA case
function symbols = lowPAPRSequence(pusch,nsc,prbset,prbrefpoint,nslot,nsymbol,ldash,symbperslot) %#ok<INUSL>
    
    if ~isempty(prbset) 
        [minprb,maxprb] = bounds(prbset);
        % Note that in the transform precoding case, DM-RS config type = 1, which is 6 DM-RS SC per 12 PRB SC (ever other subcarrier)
        % so nsc will nominally be 6
        mzc = nsc*(maxprb-minprb+1);      % Required (minimum) length of contiguous sequence to cover the PRB

        % Get low PAPR sequence for the DM-RS
        [u,v] = getHoppingParameters(pusch,mzc,nsymbol-ldash,nslot,symbperslot);  % Adjust symbol number for ldash (second double symbol)
        alpha = 0;
        symbols = reshape(nrLowPAPRS(u,v,alpha,mzc),nsc,[]);
        % Extract asociated DM-RS symbols for the PRB and turn into a column 
        symbols = reshape(symbols(:,prbset-minprb+1),[],1);
    else
        symbols = complex([]);
    end
end

% Get the sequence group number and sequence number within the group
% for the low-PAPR DM-RS case i.e. transform precoding/SC-FDMA enabled
function [u,v] = getHoppingParameters(pusch,mzc,nsymbol,nslot,symbperslot)

    % Get the scrambling identity for hopping DM-RS
    nrsid = pusch.NRSID;
    % Calculate u and v values
    v = 0;      % Sequence number in group
    fgh = 0;    % Group hopping part
    switch lower(pusch.GroupHopping)
        case 'enable'   % Group hopping, no sequence hopping
            cinit = floor(nrsid/30);
            fgh = mod(sum((2.^(0:7)').*nrPRBS(cinit,[8*(symbperslot*nslot+nsymbol) 8])),30);
        case 'disable'  % Sequence hopping, no group hopping
            cinit = nrsid;
            if mzc >= 72  % If sequence length is greater than 6*12 DM-RS subcarriers
                v = double(nrPRBS(cinit,[symbperslot*nslot+nsymbol 1]));
            end
        otherwise % 'neither'
    end
    u = mod(fgh+nrsid,30);   % Sequence group number
    
end

% Get the OFDM symbol numbers contain DM-RS for the PUSCH allocation duration
% The calling function should adjust the front loaded symbol depending on 
% the mapping type and expand for the double symbols
function dmrssymbols = lookupPUSCHDMRSymbols(pusch,typeB,nsymbols)        

    % lbar are the DM-RS positions (first symbol of pairs in the double symbol case)
    % but defined relative to the allocation start

    % Create static tables for mapping type (A/B) and single or double-symbol   
    % TS 38.211 tables 6.4.1.1.3-3 and 6.4.1.1.3-4, and 6.4.1.1.3-6
    persistent dmrs_add_pos dmrs_add_pos_hopping;
    if isempty(dmrs_add_pos) 

        % Additional position tables
        % Single-symbol, 0,1,2,3 *additional* symbols
        dmrs_singleA = {
            [],    [],      [],        [];  %  1 symbol duration
            [],    [],      [],        [];  %  2 symbol duration
            [],    [],      [],        [];  %  3 symbol duration
             0,     0,       0,         0;  %  4 symbol duration
             0,     0,       0,         0;  %  5 symbol duration
             0,     0,       0,         0;  %  6 symbol duration
             0,     0,       0,         0;  %  7 symbol duration
             0, [0,7],   [0,7],     [0,7];  %  8 symbol duration
             0, [0,7],   [0,7],     [0,7];  %  9 symbol duration
             0, [0,9], [0,6,9],   [0,6,9];  % 10 symbol duration
             0, [0,9], [0,6,9],   [0,6,9];  % 11 symbol duration
             0, [0,9], [0,6,9],[0,5,8,11];  % 12 symbol duration
             0,[0,11],[0,7,11],[0,5,8,11];  % 13 symbol duration
             0,[0,11],[0,7,11],[0,5,8,11];  % 14 symbol duration
        };
        dmrs_singleB = {
             [],     [],      [],       [];   %  1 symbol duration
             [],     [],      [],       [];   %  2 symbol duration
             [],     [],      [],       [];   %  3 symbol duration
             [],     [],      [],       [];   %  4 symbol duration 
              0,  [0,4],   [0,4],    [0,4];   %  5 symbol duration
              0,  [0,4],   [0,4],    [0,4];   %  6 symbol duration
              0,  [0,4],   [0,4],    [0,4];   %  7 symbol duration
              0,  [0,6], [0,3,6],  [0,3,6];   %  8 symbol duration
              0,  [0,6], [0,3,6],  [0,3,6];   %  9 symbol duration
              0,  [0,8], [0,4,8],[0,3,6,9];   % 10 symbol duration
              0,  [0,8], [0,4,8],[0,3,6,9];   % 11 symbol duration
              0, [0,10],[0,5,10],[0,3,6,9];   % 12 symbol duration
              0, [0,10],[0,5,10],[0,3,6,9];   % 13 symbol duration
              0, [0,10],[0,5,10],[0,3,6,9];   % 14 symbol duration
        }; 
        % Double-symbol, 0,1,2,3 *additional* symbol *pairs*
        dmrs_doubleA = {
            [],    [],[],[];    %  1 symbol duration
            [],    [],[],[];    %  2 symbol duration
            [],    [],[],[];    %  3 symbol duration
             0,     0,[],[];    %  4 symbol duration
             0,     0,[],[];    %  5 symbol duration
             0,     0,[],[];    %  6 symbol duration
             0,     0,[],[];    %  7 symbol duration
             0,     0,[],[];    %  8 symbol duration
             0,     0,[],[];    %  9 symbol duration
             0, [0,8],[],[];    % 10 symbol duration
             0, [0,8],[],[];    % 11 symbol duration
             0, [0,8],[],[];    % 12 symbol duration
             0,[0,10],[],[];    % 13 symbol duration
             0,[0,10],[],[];    % 14 symbol duration
        };
        dmrs_doubleB = {
            [],   [],[],[];    %  1 symbol duration
            [],   [],[],[];    %  2 symbol duration
            [],   [],[],[];    %  3 symbol duration
            [],   [],[],[];    %  4 symbol duration
             0,    0,[],[];    %  5 symbol duration
             0,    0,[],[];    %  6 symbol duration
             0,    0,[],[];    %  7 symbol duration
             0,[0,5],[],[];    %  8 symbol duration
             0,[0,5],[],[];    %  9 symbol duration
             0,[0,7],[],[];    % 10 symbol duration
             0,[0,7],[],[];    % 11 symbol duration
             0,[0,9],[],[];    % 12 symbol duration
             0,[0,9],[],[];    % 13 symbol duration
             0,[0,9],[],[];    % 14 symbol duration
        };   

        % Combined tables, indexed as tables{type,length}
        dmrs_add_pos = { dmrs_singleA, dmrs_doubleA; 
                         dmrs_singleB, dmrs_doubleB };
        
        % Frequency hopping cases (dimensioned for a max half slot hop)
        % Single symbol only, no double symbol configurations defined
        % 
        % Type A, starting symbol 2 case
        % 0 add pos (first/second hop) / 1 add pos (first/second hop) 
        dmrs_singleA_2FreqHop = {
            [],[],   [],   [];     % 1 symbol duration  
            [],[],   [],   [];     % 2 symbol duration
            [],[],   [],   [];     % 3 symbol duration
             2, 0,    2,    0;     % 4 symbol duration
             2, 0,    2,[0,4];     % 5 symbol duration
             2, 0,    2,[0,4];     % 6 symbol duration
             2, 0,[2,6],[0,4];     % 7 symbol duration
        };
        % Type A, starting symbol 3 case
        % 0 add pos (first/second hop) / 1 add pos (first/second hop) 
        dmrs_singleA_3FreqHop = {
            [],[],[],   [];     % 1 symbol duration  
            [],[],[],   [];     % 2 symbol duration
            [],[],[],   [];     % 3 symbol duration        
             3, 0, 3,    0;     % 4 symbol duration
             3, 0, 3,[0,4];     % 5 symbol duration
             3, 0, 3,[0,4];     % 6 symbol duration
             3, 0, 3,[0,4];     % 7 symbol duration
        };
        % Type B
        % 0 add pos (first/second hop) / 1 add pos (first/second hop) 
        dmrs_singleB_FreqHop = {
            0,0,    0,    0;  % 1 symbol duration
            0,0,    0,    0;  % 2 symbol duration
            0,0,    0,    0;  % 3 symbol duration
            0,0,    0,    0;  % 4 symbol duration
            0,0,[0,4],[0,4];  % 5 symbol duration
            0,0,[0,4],[0,4];  % 6 symbol duration
            0,0,[0,4],[0,4];  % 7 symbol duration
        };
        dmrs_add_pos_hopping = { dmrs_singleA_2FreqHop, dmrs_singleA_3FreqHop, dmrs_singleB_FreqHop };
    end
    
    % Different processing is required depending on whether 
    % frequency hopping is enabled or not
    dmrssymbols = [];
    if isfield(pusch,'IntraSlotFreqHopping') && strcmpi(pusch.IntraSlotFreqHopping,'enabled')
        % Get the relevant single symbol hopping table
        if pusch.DMRSLength == 1 || pusch.DMRSAdditionalPosition <= 1  % No DM-RS for double symbols or addpos > 1 defined
            positionstable = dmrs_add_pos_hopping{ ~typeB*(pusch.DMRSTypeAPosition-1) + typeB*3 };
            % Get the hop duration dependent symbol DM-RS position information
            n1 = floor(nsymbols/2); % Number of symbols in first hop
            n2 = nsymbols - n1;     % Number of symbols in second hop
            % First/second hop positions, defined relative to start of each hop
            if n1
                pos1 = positionstable{n1,2*pusch.DMRSAdditionalPosition+1};
            else
                pos1 = 0;   % Degenerate case of no first hop
            end
            pos2 = positionstable{n2,2*pusch.DMRSAdditionalPosition+2};
            dmrssymbols = [pos1,n1+pos2];  % Combine and adjust second position
        end
    else
        % Otherwise get the relevant non-hopping table
        positionstable = dmrs_add_pos{ typeB+1, pusch.DMRSLength };
        % Get the duration dependent symbol DM-RS position information
        if pusch.DMRSAdditionalPosition < size(positionstable,2)
            dmrssymbols = positionstable{nsymbols,1+pusch.DMRSAdditionalPosition};
        end
    end
    
end
