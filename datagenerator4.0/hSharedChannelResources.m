%hSharedChannelResources 5G NR PDSCH/PUSCH and DM-RS resource element indices and DM-RS values
%   [IND,DMRSIND,DMRS,INFO,GINFO] = hSharedChannelResources(CFT,GNB,CHS) 
%   returns the resource element (RE) indices for a 5G NR PDSCH/PUSCH
%   transmission and associated DM-RS, given the time (symbols) 
%   and frequency (PRBs) allocation of the channels and the DM-RS 
%   configuration. This function implements aspects common to both PDSCH 
%   and PUSCH, and supports OFDM based transmission without any uplink 
%   extensions to cover SC-FDMA or frequency hopping. It is used by 
%   hPDSCHResources and hPUSCHResources to provide the overall PDSCH and 
%   PUSCH resourcing and DM-RS functionality.
% 
%   The 1-based linear indices are returned in matrix IND. They are
%   defined relative to a three-dimensional RE grid representing a 
%   14/12-symbol slot for the full carrier or bandwidth (in the PDSCH/PUSCH
%   numerology) across the layers/DM-RS ports of the transmission. Each
%   column of IND represents the grid locations for a separate layer/port
%   (the third dimension of the grid). The DM-RS RE indices have the same
%   format and are returned in matrix DMRSIND. The complex values of DM-RS
%   sequence are also returned in matrix DMRS. 
%  
%   The callback configuration input, CFT, must be a structure including
%   the fields:
%   MappingType      - Slot mapping parameter name 
%                      ('PDSCHMappingType','PUSCHMappingType')
%   dmrsSymbolsTable - Function handle to provide l^bar DM-RS position indices
%   getDMRSSequence  - Optional. Function handle to override the generation
%                      of the full set of DM-RS sequence values
%                      (if parameter field is not provided or if empty then
%                      PDSCH/PUSCH PRBS based DM-RS for CP-OFDM are created)
% 
%   The general settings input, GNB, must be a structure including
%   the fields:
%   NRB             -   Number of resource blocks for the carrier
%                       or bandwidth part (in the PDSCH/PUSCH numerology)
%   CyclicPrefix      - Optional. Cyclic prefix length 
%                       ('Normal'(default),'Extended')
%   SubcarrierSpacing - Optional. Subcarrier Spacing (kHz)
%                       (15(default),30,60,120,240)
%
%   The channel specific input, CHS, must be a structure including the
%   fields:
%   NSlot                   - Optional. Slot number of transmission (default 0) 
%   PRBSet                  - PRBs allocated to the channel (0-based indices)
%   PRBRefPoint             - Optional. PRB index that PRBSet is relative to
%                             (0-based) (default 0)
%   SymbolSet               - OFDM symbols allocated to transmission within slot
%                             (including DM-RS symbols, 0-based indices, max range 0...13)
%   PortSet                 - DM-RS ports used by PDSCH/PUSCH
%                             (0-based indices, max range 0...11,
%                             mapping to ports p=1000...1011/p=0...11 respectively)
%   NLayers                 - Optional. Number of layers
%                             (Only used if PortSet has not been defined)
%   Modulation              - Modulation type(s) of codeword(s)
%                             ('pi/2-BPSK','BPSK','QPSK','16QAM','64QAM','256QAM')
%   Reserved                - Optional. Reserved PRB patterns
%                             (structure array, see below)
%   DMRSConfigurationType   - DM-RS configuration type (1,2)
%   NumCDMGroupsWithoutData - Optional. Number of CDM groups without data
%                             (0(default),1,2,3)
%
%   If FT.getDMRSSequence does not override the generation of the DM-RS
%   values then the following scrambling identity parameters are required:
%   NIDNSCID                - DM-RS scrambling identity (0...65535)
%   NSCID                   - DM-RS scrambling initialization (0,1)
% 
%   The DM-RS OFDM symbol locations can either be defined explicitly 
%   using the single parameter:
%   DMRSSymbolSet           - OFDM symbols containing the DM-RS within the 
%                             shared channel allocation in slot 
%                             (0-based indices, max range 0...13)
%   or, defined implicitly via the FT.dmrsSymbolsTable function handle and 
%   its own parameter requirements, and the following DM-RS parameters: 
%   DMRSTypeAPosition       - Mapping type A only. First DM-RS symbol position
%                             (2,3)
%   DMRSLength              - Number of front-loaded DM-RS symbols
%                             (1(single symbol),2(double symbol))
%
%   Periodically recurring patterns of reserved PRB can defined using the
%   'Reserved' parameter. These PRB will be excluded from the generated 
%   indices and the transport and physical processing should rate-match 
%   around them. This parameter takes the format of a structure array where
%   each element defines a separate pattern. Each element, i, of the 
%   array should contain the following fields:
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
%   signaled configuration types ('dmrs_Type'). Configuration
%   type 1 defines 6 subcarriers per PRB per antenna port, comprising
%   alternating subcarriers. Configuration type 2 defines 4 subcarriers per
%   PRB per antenna ports, consisting of 2 groups of 2 neighboring
%   subcarriers. Different shifts are applied to the sets of subcarriers
%   used, dependent on the associated antenna port or CDM group. For
%   type 1, there are 2 possible CDM groups/shifts across up to 8 possible
%   antenna ports (p=0...11 for uplink/p=1000...1007 for downlink), and, 
%   for type 2, there are 3 possible CDM groups/shifts across 12 ports
%   (p=0...11/p=1000...1011). See TS 38.211 sections 6.4.1.1 and 7.4.1.1
%   for the full configuration details.
% 
%   INFO is the output structure containing the fields:
%   G             - Bit capacity of the PDSCH/PUSCH. This is should be the
%                   length of codeword to be output from the DL-SCH/UL-SCH 
%                   transport channel
%   Gd            - Number of resource elements per layer/port, equal to 
%                   the number of rows in the PDSCH/PUSCH indices
%   DMRSSymbolSet - The symbol numbers in a slot containing DM-RS (0-based)
%   NREPerPRB     - Number of RE per PRB allocated to PDSCH/PUSCH (not 
%                   accounting for any reserved resources)
%   CDMGroups     - CDM groups associated with the DM-RS antenna ports
%
%   GINFO is the output structure containing the fields:
%   SymbolsPerSlot - Number of OFDM symbols in a slot, 14 or 12 for normal
%                    and extended cyclic prefix, respectively
%
%   Example:
%   % Configure the function to create PRBS based DM-RS for 3 OFDM symbols 
%   % in a slot. Display the associated data channel and DM-RS resource
%   % element locations.
% 
%   % Define a callback configuration structure
%   ft = struct();
%   % Mapping type parameter name
%   ft.MappingType = 'PDSCHMappingType';    
%   % As a simple example, set up a function handle which will always
%   % returns 3 DM-RS position indices 0, 5 and 10.
%   ft.dmrsSymbolsTable = @(pxsch,typeb,nsymbols)[0 5 10];
%   % Don't override the default PRBS based DM-RS sequence generation
%   ft.getDMRSSequence = [];
%
%   % Set the number of resource blocks in carrier or BWP and 
%   % the numerology
%   gnb = struct('NRB',50);
%   gnb.SubcarrierSpacing = 15;
%   gnb.CyclicPrefix = 'Normal';
% 
%   % Get the number of OFDM symbols in a slot
%   symbperslot = sum(strcmpi(gnb.CyclicPrefix,["Normal","Extended"]) .* [14 12]);
%
%   % Specify the basic allocation properties to be full band, full
%   % slot using 2 layers/ports
%   pxsch = struct();
%   pxsch.NSlot = 0;                    % Slot number
%   pxsch.PRBSet = 0:gnb.NRB-1;         % Full band PRB allocation
%   pxsch.SymbolSet = 0:symbperslot-1;  % Full slot allocation
%   pxsch.PortSet = [0,2];              % Use DM-RS ports 1000 and 1002
%   pxsch.Modulation = 'QPSK';          % Modulation type
%
%   % Exclude some PRBs (e.g. for a CORESET in the downlink) in the middle
%   % of the carrier in the first two symbols in the slot
%   pxsch.Reserved.PRB = fix(gnb.NRB/2) + (-4:5);
%   pxsch.Reserved.Symbols = [0,1];
%   pxsch.Reserved.Period = 1;
% 
%   % Configure the DM-RS for config type 1 and slot-wise, type A
%   % mapping. Use double symbols for front-loaded DM-RS and configure an
%   % additional symbol pair towards the end of the slot
%   pxsch.DMRSConfigurationType = 1; % DM-RS configuration type (1,2)
%   pxsch.NIDNSCID = 1;            % DM-RS scrambling identity (0...65535)
%   pxsch.NSCID = 0;               % DM-RS scrambling initialization (0,1)
%   pxsch.PDSCHMappingType = 'A';  % Slot-wise mapping type (type A)
%   pxsch.DMRSTypeAPosition = 2;   % First DM-RS symbol position for type A 
%   pxsch.DMRSLength = 1;          % Specify single front-loaded DM-RS
%
%   % Display data and DM-RS RE locations of the first port of the grid
%   slotgrid = zeros(12*gnb.NRB,symbperslot,length(pxsch.PortSet));
%   [ind,dmrsind,dmrs,info] = hSharedChannelResources(ft,gnb,pxsch);
%   slotgrid(ind) = 20;                 % Use light blue for data RE 
%   slotgrid(dmrsind) = 50*abs(dmrs);   % Use yellow for DM-RS RE
%   figure;
%   image(slotgrid(:,:,1));
%   title('Data and DM-RS resource elements');
%   axis('xy'), xlabel('Symbols'), ylabel('Subcarriers');
%  
%   See also hPDSCHResources, hPUSCHResources.

%   Copyright 2019 The MathWorks, Inc.
 
function [pxschIndices,dmrsIndices,dmrSymbols,pxschInfo,genInfo] ...
                                      = hSharedChannelResources(ftable,gnb,pxsch)

    % Capture the SCS and number of slots in a 10ms frame
    if isfield(gnb,'SubcarrierSpacing')
        scs = gnb.SubcarrierSpacing;
    else
        scs = 15;  % Default to 15kHz SCS
    end
    slotsperframe = 10*(scs/15);
    
    % Establish the number of OFDM symbols in a single slot
    if isfield(gnb,'CyclicPrefix')
        cpoptions = ["Normal","Extended"];
        cpselected = strcmpi(gnb.CyclicPrefix,cpoptions);
        if ~any(cpselected)
            error('The cyclic prefix (%s) must be one of (%s)' , string(gnb.CyclicPrefix), join(cpoptions,','));
        end
        symbperslot = sum(cpselected .* [14 12]);
    else
        symbperslot = 14;   % Default to normal CP
    end
    
    % Capture the set of active OFDM symbols (within the slot) associated
    % with the allocation
    if isfield(pxsch,'SymbolSet')
        usedsymbols = pxsch.SymbolSet < symbperslot;
        allocatedsymbols = pxsch.SymbolSet(usedsymbols);        
    else
        usedsymbols = ones(1,symbperslot);
        allocatedsymbols = (0:symbperslot-1);   % Default to a full slot allocation
    end
   
    % Capture set of PRB (within the slot) associated with the allocation
    if iscell(pxsch.PRBSet)
        if length(pxsch.PRBSet) ~= length(usedsymbols)
            chname = extractBefore(ftable.MappingType,'Mapping');
            error('When PRBSet is a cell array, its length (%d) must be the same as the number of OFDM symbols defined for the %s allocation (%d).',...
                       length(pxsch.PRBSet),chname,length(usedsymbols));
        end
        allocatedPRB =  pxsch.PRBSet(usedsymbols);
    else
        allocatedPRB = pxsch.PRBSet;
    end
    
    % Capture the set of antenna ports associated with the DM-RS
    % Config type 1: p=(10)00-(10)07, Config type 2: p=(10)00-(10)11 (max ranges for PDSCH/PUSCH)
    if isfield(pxsch,'PortSet')
        ports = reshape(pxsch.PortSet,1,[]);
    else
        if isfield(pxsch,'NLayers') 
            ports = 0:pxsch.NLayers-1;     % Default to number of tx layers, if provided
        else
            ports = 0;                     % Otherwise single, single port
        end
        pxsch.PortSet = ports;
    end    
    
    % Capture the set of OFDM symbols (within the slot) that will carry the DM-RS
    % If not explicitly defined then calculate DM-RS containing OFDM symbols from parameter set
    if isfield(pxsch,'DMRSSymbolSet')
        dmrssymbols = intersect(pxsch.DMRSSymbolSet,allocatedsymbols);
        ldash = zeros(1,length(dmrssymbols));   % In the case, treat as all single symbols
    else
        % Establish whether it's type A or type B mapping
        mp = strcmpi(pxsch.(ftable.MappingType),["A","B"]);
        if ~any(mp)
            error('%s (%s) must be A or B.',ftable.MappingType,pxsch.(ftable.MappingType));
        end
        typeB = mp(2);
        [dmrssymbols,ldash] = getDMRSSymbolIndices(ftable.dmrsSymbolsTable,pxsch,typeB,allocatedsymbols);
    end
    
    % Detect and warn when no DM-RS symbols are defined for the input setup
    if ~isempty(allocatedsymbols) && isempty(dmrssymbols)
        chname = extractBefore(ftable.MappingType,'Mapping');
        warning('nr5g:DMRSParametersNoSymbols','No DM-RS symbols are defined for the given set of DM-RS configuration parameters and %s resource allocation. Cross-check with the %s DM-RS position tables.',...
                        chname,chname);
    end
    
    % Capture the slot number of the PDSCH/PUSCH
    % Used to locate the current slot in relation to any extended reserved resource patterns
    % and for the DM-RS sequence generation
    if ~isfield(pxsch,'NSlot')
       pxsch.NSlot = 0;     % Default to 0 if not present
    end

    % Capture the number of CDM groups without data
    if isfield(pxsch,'NumCDMGroupsWithoutData')
       cdmgroupsnodata = pxsch.NumCDMGroupsWithoutData;
       if ~any(cdmgroupsnodata == 0:3)
           error('NumCDMGroupsWithoutData (%d) must be one of (0,1,2,3).',cdmgroupsnodata);
       end
    else
       cdmgroupsnodata = 0;
    end
    
    % PRB level mapping (PRB carrying the PDSCH/PUSCH)
    % Find all allocated PRB (accounting for any reserved PRB) across
    % the allocated OFDM symbols
    
    % Create a cell array containing the PRB set used in each allocated OFDM symbol
    prbcell = cell(1,symbperslot);
    if iscell(allocatedPRB)
        prbcell(allocatedsymbols+1) = cellfun(@(x)reshape(mod(x,gnb.NRB),1,[]),allocatedPRB,'UniformOutput',false);
    else
        prbset = reshape(mod(allocatedPRB,gnb.NRB),1,[]);        % Ensure that it is now a row for implicit expansion later
        prbcell(allocatedsymbols+1) = {prbset};
    end
       
    % Loop over the reserved resources and exclude them from the allocation PRB
    % across the allocated symbols, stored in prbcell
    if isfield(pxsch,'Reserved')
        for ri=1:length(pxsch.Reserved)

            % Reference the current reserved symbol/PRB indices pair and period
            reservedsymbols = pxsch.Reserved(ri).Symbols;
            reservedprb = reshape(pxsch.Reserved(ri).PRB,1,[]);
            reservedperiod = pxsch.Reserved(ri).Period;

            % Find any of the allocated symbols which overlap with reservations
            %
            % If the reserved period was empty then get number of complete slots
            % in reservation period and cyclically extend pattern to cover current slot 
            if isempty(reservedperiod)                
                reservedperiod = 0;
            end
            offset = mod(pxsch.NSlot,reservedperiod)*symbperslot;        % Symbol offset (whole number of slots) into pattern period                       
            inter = intersect(allocatedsymbols,reservedsymbols-offset);  % Find allocated symbols in current slot which contain reserved PRB
            % Reference the PRB associated with the overlapping symbols
            if iscell(reservedprb)
                if length(reservedprb) ~= length(reservedsymbols)  
                    error('Symbol-wise reserved PRB resources (defined in a cell array) and associated symbols should have the same number of elements.');
                end
                prbdiff = arrayfun(@(x)setdiff(prbcell{x+1},[reservedprb{x==(reservedsymbols-offset)}]), inter,'UniformOutput',false);
            else
                prbdiff = arrayfun(@(x)setdiff(prbcell{x+1},reservedprb), inter,'UniformOutput',false);
            end
            prbcell(inter+1) = prbdiff;
        end   
    end

    % RE level mapping (RE per PRB carrying the PDSCH/PUSCH)
    % Find all PDSCH/PUSCH RE per PRB for each OFDM symbol that will also carry DM-RS

    if ~any(pxsch.DMRSConfigurationType==[1,2])
        error('DMRSConfigurationType (%d) must be either 1 or 2.',pxsch.DMRSConfigurationType);
    end
    if pxsch.DMRSConfigurationType==1
        % Type 1: 6 DM-RS SC per PRB per CDM (every other SC)
        dmrssc = [0 2 4 6 8 10]';                   % RE indices in a PRB
        cdmgroups = mod(fix(ports/2),2);            % CDM groups associated with the used ports
        dshifts = cdmgroups;                        % Delta shift per used CDM group (0,1)... the *set* of shifts/groups that we will exclude
        dshiftsnodata = 0:min(cdmgroupsnodata,2)-1; % Delta shifts for CDM groups without data
    else      
        % Type 2: 4 DM-RS SC per PRB per CDM (2 groups of 2 SC)
        dmrssc = [0 1 6 7]';                            % RE indices in a PRB
        cdmgroups = mod(fix(ports/2),3);                % CDM groups associated with the used ports
        dshifts = 2*cdmgroups;                          % Delta shift per used CDM group (0,2,4)... the *set* of shifts/groups that we will exclude  
        dshiftsnodata = 2*(0:min(cdmgroupsnodata,3)-1); % Delta shifts for CDM groups without data
    end

    fullprb = ones(12,1);        % Binary map of all the subcarriers in a PRB
    dmrsre = dmrssc + [dshifts dshiftsnodata];% Implicit expansion of zero-shifted DM-RS RE positions (col) across all the shifts/CDMs (row) in play   
    fullprb(dmrsre+1) = 0;       % Clear all RE which will carry DM-RS in at least one port
    pxschre = find(fullprb)-1;   % Find PDSCH/PUSCH (non DM-RS) RE in a DM-RS containing symbol
    
    % Create the RE per PRB list across the allocated symbols, accounting
    % for DM-RS and non DM-RS carrying cases
    recell = cell(1,symbperslot);
    recell(allocatedsymbols+1) = {(0:11)'};  % Data RE per PRB in normal data symbols
    recell(dmrssymbols+1) = {pxschre};       % Data RE per PRB in DM-RS carrying symbols
      
    % Combine PRB oriented and RE per PRB oriented index arrays and expand (using implicit expansion)
    % into a column of linear indices for a single antenna/layer
    % Implicit expansion:                         Column             Row                                      Column
    slotindices = cell2mat( arrayfun(@(x) reshape(recell{x+1} + 12*prbcell{x+1} + 12*gnb.NRB*x,[],1), allocatedsymbols(:),'UniformOutput',false) );
     
    % Expand across all antenna ports/layers and make 1-based
    % Implicit expansion: Column                                        Row     
    pxschIndices = slotindices(:) + (1 + (12*symbperslot*gnb.NRB)*(0:numel(ports)-1));
    
    % Channel bit capacity information
    nports = numel(ports);                      % Numbers of ports/layers across all codewords
    ncw = 1 + (nports > 4);                     % Number of codewords, deduced from total layers
    nlayers = fix((nports + (0:ncw-1))/ncw);    % Number of layers per codeword
    pxschInfo.Gd = size(pxschIndices,1);        % Number of QAM data symbols in one layer/antenna grid
    
    % Validate modulation type and translate into bits per symbol
    fullmodlist = ["pi/2-BPSK","BPSK","QPSK","16QAM","64QAM","256QAM"]'; % All NR mod schemes
    modulation = reshape(string(pxsch.Modulation),1,[]);              % Turn into a string row
    qm = [1 1 2 4 6 8]*(lower(modulation) == lower(fullmodlist));     % Test each element against all mods then multiply against bps
    if ~all(qm)
        error("The modulation (%s) must be one of the set (%s).",join(modulation(logical(qm == 0))), join(fullmodlist',','));
    end
    pxschInfo.G = qm.*pxschInfo.Gd.*nlayers;     % And scale by layers/ports and modulation scheme
    pxschInfo.NREPerPRB  = 12*(length(allocatedsymbols)-length(dmrssymbols)) + length(pxschre)*length(dmrssymbols);
  
    % Additional DM-RS related information
    pxschInfo.DMRSSymbolSet = dmrssymbols;      % OFDM symbols containing DM-RS
    pxschInfo.CDMGroups = cdmgroups;            % CDM group associated with each port
  
    % Create the DM-RS QPSK symbols and resource element indices associated
    % with the shared transmission
    %
    % First create *unshifted* version of the DM-RS RE indices in a single antenna plane
    % Implicit expansion:                           Column        Row          Scalar offset          Column
    dmslotindices = cell2mat( arrayfun(@(x) reshape(dmrssc + 12*prbcell{x+1} + 12*gnb.NRB*x,[],1), dmrssymbols(:),...
                              'UniformOutput',false) );
    
    % Expand across all antenna ports/layers, applying DM-RS shifts for each port, and make 1-based
    % Implicit expansion: Column             Row          Scalar Offset          Row
    dmrsIndices = dmslotindices(:) + (1 + dshifts + (12*symbperslot*gnb.NRB)*(0:nports-1));
    
    % If an overriding DM-RS sequence generator is defined then use it
    % otherwise use the local NIDNSCID/NSCID PRBS based generator
    if ~isfield(ftable,'getDMRSSequence') || isempty(ftable.getDMRSSequence)
        ftable.getDMRSSequence = @prbsDMRSSequence;
    end
    % Create the matching set of DM-RS QPSK symbols for all ports
    % First calculate the frame relative slot number from the (absolute) input slot number
    nslot = mod(pxsch.NSlot,slotsperframe);
    dmrSymbols = getDMRSValues(ftable.getDMRSSequence,pxsch,prbcell(dmrssymbols+1),nslot,dmrssymbols,ldash,symbperslot);
    
    % Additional structural information to be passed back to the calling function
    genInfo.SymbolsPerSlot = symbperslot;
end

% Get the OFDM symbol indices containing DM-RS for the PDSCH/PUSCH allocation
% Uses a function handle callback to look up the initial l^bar position values
% which are then adjusted for the mapping type and duration (single/double)
function [dmrssymbols,ldash] = getDMRSSymbolIndices(gettablesymbols,pxsch,typeb,allocatedsymbols)  
    
    % Check DMRSLength value for later use
    dmslength = pxsch.DMRSLength;
    if ~isscalar(dmslength) || ~any(dmslength == [1 2])
        error('DMRSLength (%s) must be either 1 or 2.', join(string(dmslength),','));
    end

    % Get PDSCH/PUSCH allocation duration, l_d  
    [lb,ub] = bounds(allocatedsymbols);
    if ~typeb
        lb = 0;
    end
    nsymbols = ub - lb + 1;
    
    if nsymbols
        % Get the OFDM symbol indices, l^bar, that will carry DM-RS for the channel instance
        dmrssymbols = gettablesymbols(pxsch,typeb,nsymbols);
    else
        dmrssymbols = [];
    end
    
    % l' values associated with DM-RS symbol indices
    ldash = zeros(1,length(dmrssymbols)*dmslength);
    % Adjust table information
    if ~isempty(dmrssymbols)
        % Adjust indices for the relative offset of the mapping type
        if typeb
           dmrssymbols = dmrssymbols+allocatedsymbols(1);   % If type B (non slot-wise)
        else
           dmrssymbols(1) = pxsch.DMRSTypeAPosition;        % If type A (slot-wise) then 2 or 3
        end
        % Adjust for double-symbol DM-RS
        % In the operational system, if RRC configured with max length 2
        % then the actual length is DCI signaled
        if pxsch.DMRSLength == 2
            dmrssymbols = reshape([dmrssymbols; dmrssymbols+1],1,[]);
            ldash(2:2:end) = 1;
        end
        % For non-standard set-ups, only return the DM-RS symbol indices that 
        % overlap the actual allocation indices
        [dmrssymbols,~,indices] = intersect(allocatedsymbols,dmrssymbols);
        ldash = ldash(indices);
    end
end

% Get all complex symbol values associated with the DM-RS for the transmission
% Uses a function handle callback to 
% Each separate antenna port is a column of the returned matrix which concatenates
% all the DM-RS values for all DM-RS carrying OFDM symbols associated with transmission
function symb = getDMRSValues(getDMRSSequence,pxsch,prbset,nslot,dmrssymbols,ldashvalues,symbperslot)
       
    % Capture the PRB/subcarrier reference point associated with PRB set values, 
    % for the DM-RS sequence indexing
    if isfield(pxsch,'PRBRefPoint')
       prbrefpoint = pxsch.PRBRefPoint;
    else
       prbrefpoint = 0;
    end
    
    % Capture set of antenna ports required
    ports = pxsch.PortSet;
    
    % 6 DM-RS QPSK symbols (type 1) or 4 DM-RS QPSK symbols (type 2) per PRB
    ndmrsre = 4 + (pxsch.DMRSConfigurationType==1)*2;     
    
    % Loop over the OFDM symbols containing DM-RS 
    symcell = cell(1,length(dmrssymbols));
    for i=1:length(dmrssymbols)
        symcell{i} = reshape(getDMRSSequence(pxsch,ndmrsre,prbset{i},prbrefpoint,nslot,dmrssymbols(i),ldashvalues(i),symbperslot),1,[]);
    end

    % Accumulated lengths of the DM-RS when concatenated across the OFDM symbols 
    cndmrs = [0 cumsum(cellfun(@length,symcell))];
    % Preallocate array for the returned DM-RS
    symb = zeros(cndmrs(end),numel(ports));

    % Establish the max number of DM-RS ports, depending on the config type
    maxnports = 8+(pxsch.DMRSConfigurationType==2)*4;
    % Loop over the ports and apply masks
    for pidx = 1:numel(ports)        
        % Mask applied if (p is odd (f part)) or (p >= (100)4 (type 1) or (100)6 (type 2) and double-symbol (t part))
        % Test for 'f' part of mask (subcarrier, frequency part)
        pv = ports(pidx);
        if mod(pv,2)
          fmask = [1 -1];
        else
          fmask = 1;
        end
        % Test for the 't' part, based on the l' values
        tmask = 1-2*(ldashvalues*(pv >= maxnports/2));
        % Apply combined time and frequency mask values and concatenate 
        % across all the OFDM symbols
        for i = 1:length(symcell)             
            symb(cndmrs(i)+1:cndmrs(i+1),pidx) = symcell{i}.*repmat(tmask(i)*fmask,size(symcell{i})./size(fmask));
        end
    end
end

% Generate PRBS based DM-RS sequence for a single OFDM symbol
function symbols = prbsDMRSSequence(pxsch,ndmrssc,prbset,prbrefpoint,nslot,nsymbol,ldash,symbperslot) %#ok<INUSL>
    
    % Cache the scrambling IDs
    nidnscid = pxsch.NIDNSCID;
    nscid = pxsch.NSCID;
    
    if ~isempty(prbset)
        % Generate PRBS for DM-RS sequence which covers the PRB allocation set range 
        [minprb,maxprb] = bounds(prbset);
        cinit = mod(2^17*(symbperslot*nslot + nsymbol + 1)*(2*nidnscid + 1) + 2*nidnscid + nscid,2^31);
        prbs = reshape(nrPRBS(cinit,2*ndmrssc*[prbrefpoint+minprb maxprb-minprb+1]),2*ndmrssc,[]);
    
        % Extract PRBS values associated with PRB and turn into complex DM-RS symbols
        bpsk = 1/sqrt(2)*(1-2*reshape(prbs(:,prbset-minprb+1),2,[])');
        symbols = complex(bpsk(:,1),bpsk(:,2));
    else
        symbols = complex([]);
    end
end   
