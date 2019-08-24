%
% Programmed by Chanwoo Kim for the IEEETran Speech, Audio, and Langauge Processing and ICASSP 2012
%
% (chanwook@cs.cmu.edu)
%
% Important: The input should be mono and be sampled at "16 kHz".
%
%
% * If you want to use logarithmic nonlinearity instead of the power
% nonlinearity, change bPowerLaw to 0 (lilne 28)
%
% PNCC_IEEETran(OutFile, InFile)
%
% iFiltTyep 2 Gamma
% iFiltType 1 HTK MEL
% iFiltType 0 Snaley MEL
% Default
% 0.5, 0.01, 2, 4
%
function [aadDCT] = pncc(ad_x)

	dLamda_L = 0.999;
	dLamda_S = 0.999;

	dSampRate   = 16000;
	dLowFreq      = 200;
	dHighFreq     = dSampRate / 2;
	dPowerCoeff = 1 / 15;

	iFiltType = 2;
	dFactor = 2.0;

	dGammaThreshold = 0.005;

	iM = 2;
	iN = 4;

	iSMType = 0;
    
	dLamda  = 0.999;
	dLamda2 = 0.5;
	dDelta1 = 1;

	dLamda3 = 0.85;
	dDelta2 = 0.2;
  
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	% Flags
       %
	bPreem         = 1;
	bSSF             = 1;
	bPowerLaw    = 1;
	bDisplay        = 0;
     

	dFrameLen     = 0.0256;  % 25.6 ms window length, which is the default setting in CMU Sphinx
	dFramePeriod = 0.010;   % 10 ms frame period
	iPowerFactor  = 1;

	iFFTSize  = 1024;
	iNumFilts = 40;
       
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	% Flags
	%
	%
	% Array Queue Ring-buffer
	%
	global Queue_aad_P;
	global Queue_iHead;
	global Queue_iTail;
	global Queue_iWindow;
	global Queue_iNumElem;

	Queue_iWindow  = 2 * iM + 1;
	Queue_aad_P    = zeros(Queue_iWindow, iNumFilts);
	Queue_iHead    = 0;
	Queue_iTail    = 0;
	Queue_iNumElem = 0;
   
	iFL        = floor(dFrameLen    * dSampRate);
	iFP        = floor(dFramePeriod * dSampRate);
	iNumFrames = floor(length(ad_x) / iFP);
	iSpeechLen = length(ad_x);
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	% Pre-emphasis using H(z) = 1 - 0.97 z ^ -1
	%
	if (bPreem == 1)
	    ad_x = filter([1 -0.97], 1, ad_x);
	end

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	% Obtaning the gammatone coefficient. 
	%
	% Based on M. Snelly's auditory toolbox. 
	% In actual C-implementation, we just use a table
	%
	bGamma = 1;
    
	if (iFiltType == 2)
	aad_H = ComputeFilterResponse(iNumFilts, iFFTSize, dLowFreq, dHighFreq, dSampRate);
	aad_H = abs(NormalizeFilterGain(aad_H, dSampRate));
	for i = 1 : iNumFilts,
	    aiIndex = find(aad_H(:, i) < dGammaThreshold * abs(max(aad_H(:, i))));
	    aad_H(aiIndex, i) = 0;
	end

	else
	    [wts,binfrqs]  = fft2melmx(iFFTSize, dSampRate, iNumFilts, 1, dLowFreq, dHighFreq, iFiltType);
	    wts = wts';
	    wts(size(wts, 1) / 2 + 1 : size(wts, 1), : ) = [];
	   aad_H = wts;
	    
	end
    
	i_FI     = 0;
	i_FI_Out = 0;

	if bSSF == 1
	    adSumPower = zeros(1, iNumFrames - 2 * iM);
	else
	    adSumPower = zeros(1, iNumFrames);
	end
	 
	%dLamda_L   = 0.998;
	aad_P      = zeros(iNumFrames,      iNumFilts);
	aad_P_Out  = zeros(iNumFrames - 2 * iM,      iNumFilts);
	ad_Q       = zeros(1,               iNumFilts);
	ad_Q_Out   = zeros(1,               iNumFilts);
	ad_QMVAvg  = zeros(1,               iNumFilts);
	ad_w       = zeros(1,               iNumFilts);
	ad_w_sm    = zeros(1,               iNumFilts);
	ad_QMVAvg_LA = zeros(1,               iNumFilts);

	MEAN_POWER = 1e10;

	dMean  = 5.8471e+08;
	dPeak = 2.7873e+09 / 15.6250;
	% (1.7839e8, 2.0517e8, 2.4120e8, 2.9715e8, 3.9795e8) 95, 96, 97, 98, 99
	% percentile from WSJ-si84
	            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	dPeakVal = 4e+07;% % 4.0638e+07  --> Mean from WSJ0-si84  (Important!!!)
	                %%%%%%%%%%%
	dMean = dPeakVal;
    
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	% Obtaining the short-time Power P(i, j)
	%
	for m = 0 : iFP : iSpeechLen  - iFL 
		ad_x_st                = ad_x(m + 1 : m + iFL) .* hamming(iFL);
		adSpec                 = fft(ad_x_st, iFFTSize);
		ad_X                   = abs(adSpec(1: iFFTSize / 2));
		aadX(:, i_FI + 1)      = ad_X; 
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%
		% Calculating the Power P(i, j)
		%
		for j = 1 : iNumFilts
		        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		        %
		        % Squared integration
		        %
		        
		        if iFiltType == 2
		            aad_P(i_FI + 1, j)  = sum((ad_X .* aad_H(:, j)) .^ 2);
		        else
		            aad_P(i_FI + 1, j)  = sum((ad_X .^ 2 .* aad_H(:, j)));
		        end
		        
		end
        
	        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	        %
	        % Calculating the Power P(i, j)
	        %
	        
	        dSumPower = sum(aad_P(i_FI + 1, : ));
	             
	        if bSSF == 1
			%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			%
			% Ring buffer (using a Queue)
			%
			if (i_FI >= 2 * iM + 1)
				Queue_poll();
				end
				Queue_offer(aad_P(i_FI + 1, :));

				ad_Q = Queue_avg();

			if (i_FI == 2 * iM)
				ad_QMVAvg     = ad_Q.^ (1 / 15);
				ad_PBias  =  (ad_Q) * 0.9;
			end
          
			if (i_FI >= 2 * iM)  
				%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				%
				% Bias Update
				%
				for i = 1 : iNumFilts,
				    if (ad_Q(i) > ad_PBias(i))
				       ad_PBias(i) = dLamda * ad_PBias(i)  + (1 - dLamda) * ad_Q(i);
				    else
				       ad_PBias(i) = dLamda2 * ad_PBias(i) + (1 - dLamda2) * ad_Q(i);
				    end
				end
			            
			       for i = 1 : iNumFilts,
				ad_Q_Out(i) =   max(ad_Q(i) - ad_PBias(i), 0) ;

				if (i_FI == 2 * iM)
				    ad_QMVAvg2(i)  =  0.9 * ad_Q_Out(i);
				    ad_QMVAvg3(i)  =  ad_Q_Out(i);
				    ad_QMVPeak(i)  =  ad_Q_Out(i);
				end

				if (ad_Q_Out(i) > ad_QMVAvg2(i))
				     ad_QMVAvg2(i) = dLamda * ad_QMVAvg2(i)  + (1 -  dLamda)  *  ad_Q_Out(i);
				else
				     ad_QMVAvg2(i) = dLamda2 * ad_QMVAvg2(i) + (1 -  dLamda2) *  ad_Q_Out(i);
				end

				dOrg =  ad_Q_Out(i);

				ad_QMVAvg3(i) = dLamda3 * ad_QMVAvg3(i);
			          
				if (ad_Q(i) <  dFactor * ad_PBias(i))
				    ad_Q_Out(i) = ad_QMVAvg2(i);
				else
				     if (ad_Q_Out(i) <= dDelta1 *  ad_QMVAvg3(i))
				        ad_Q_Out(i) = dDelta2 * ad_QMVAvg3(i);
				     end
				end
				ad_QMVAvg3(i) = max(ad_QMVAvg3(i),   dOrg);

				ad_Q_Out(i) =  max(ad_Q_Out(i), ad_QMVAvg2(i));
			end
			      ad_w      =   ad_Q_Out ./ max(ad_Q, eps);

			for i = 1 : iNumFilts,
				 if iSMType == 0
			                ad_w_sm(i) = mean(ad_w(max(i - iN, 1) : min(i + iN ,iNumFilts)));
			        elseif iSMType == 1
			                ad_w_sm(i) = exp(mean(log(ad_w(max(i - iN, 1) : min(i + iN ,iNumFilts)))));
			        elseif iSMType == 2
			        	 ad_w_sm(i) = mean((ad_w(max(i - iN, 1) : min(i + iN ,iNumFilts))).^(1/15))^15;
			        elseif iSMType == 3
			 		ad_w_sm(i) = (mean(  (ad_w(max(i - iN, 1) : min(i + iN ,iNumFilts))).^15 )) ^ (1 / 15); 
			        end
			end        
			        
		        aad_P_Out(i_FI_Out + 1, :) = ad_w_sm .* aad_P(i_FI - iM + 1, :);
		        adSumPower(i_FI_Out + 1)   = sum(aad_P_Out(i_FI_Out + 1, :));

		        if  adSumPower(i_FI_Out + 1) > dMean
		             dMean = dLamda_S * dMean + (1 - dLamda_S) * adSumPower(i_FI_Out + 1);
		        else
		             dMean = dLamda_L * dMean + (1 - dLamda_L) * adSumPower(i_FI_Out + 1);
		        end
		        
		        aad_P_Out(i_FI_Out + 1, :) = aad_P_Out(i_FI_Out + 1, :) / (dMean)  * MEAN_POWER;
		        i_FI_Out = i_FI_Out + 1;
		        
		end
           
	else % if not SSF
		adSumPower(i_FI + 1)   = sum(aad_P(i_FI + 1, :));
             
		if  adSumPower(i_FI_Out + 1) > dMean
		     dMean = dLamda_S * dMean + (1 - dLamda_S) * adSumPower(i_FI_Out + 1);
		else
		     dMean = dLamda_L * dMean + (1 - dLamda_L) * adSumPower(i_FI_Out + 1);
		end

		aad_P_Out(i_FI + 1, :) = aad_P(i_FI + 1, :) / (dMean)  * MEAN_POWER;
		end
		i_FI = i_FI + 1;
	end

	
	%adSorted  = sort(adSumPower);
	%dMaxPower = adSorted(round(0.98 * length(adSumPower)));
	%aad_P_Out = aad_P_Out / dMaxPower * 1e10;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	% Apply the nonlinearity
	%
	%dPowerCoeff
	if bPowerLaw == 1
	    aadSpec = aad_P_Out .^ dPowerCoeff;
	else
	    aadSpec = log(aad_P_Out + eps);
	end

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	% DCT
	%
	aadDCT                  = dct(aadSpec')';
	aadDCT(:, 14:iNumFilts) = [];

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	% CMN
	%
	for i = 1 : 13
	       aadDCT( :, i ) = aadDCT( : , i ) - mean(aadDCT( : , i));
	end

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	% Display
	%
	if bDisplay == 1
	    figure
	    
	    aadSpec = idct(aadDCT', iNumFilts);
	    imagesc(aadSpec); axis xy;
	end

	aadDCT = aadDCT';

    
end

function [] = Queue_offer(ad_x)
    global Queue_aad_P;
    global Queue_iHead;
    global Queue_iTail;
    global Queue_iWindow;
    global Queue_iNumElem;
    
    Queue_aad_P(Queue_iTail + 1, :) = ad_x;
    
    Queue_iTail    = mod(Queue_iTail + 1, Queue_iWindow);
    Queue_iNumElem = Queue_iNumElem + 1;
    
    if Queue_iNumElem > Queue_iWindow
       error ('Queue overflow'); 
    end
    
  
end


function [ad_x] = Queue_poll()
    global Queue_aad_P;
    global Queue_iHead;
    global Queue_iTail;
    global Queue_iWindow;
    global Queue_iNumElem;
    
   
    
    if Queue_iNumElem <= 0
       error ('No elements'); 
    end
    
    
    ad_x =  Queue_aad_P(Queue_iHead + 1, :);
    
    Queue_iHead    = mod(Queue_iHead + 1, Queue_iWindow);
    Queue_iNumElem = Queue_iNumElem - 1;
 
end


function[adMean] = Queue_avg()

    global Queue_aad_P;
    global Queue_iHead;
    global Queue_iTail;
    global Queue_iWindow;
    global Queue_iNumElem;
    
    adMean = zeros(1, 40);

    
    iPos = Queue_iHead;
    
    
    for i = 1 : Queue_iNumElem
        adMean = adMean + Queue_aad_P(iPos + 1 ,: );
        iPos   = mod(iPos + 1, Queue_iWindow);
    end
    
    adMean = adMean / Queue_iNumElem;

end
