function [notasDoAudio,acorde] = getNota(frequencia)


freqFund = 16.351597;


notasDoAudio = mod(round(12*log2(frequencia/freqFund)),12);
notasDoAudio = notasDoAudio + ones(1,length(notasDoAudio));

acorde = unique(notasDoAudio,'stable');



end



