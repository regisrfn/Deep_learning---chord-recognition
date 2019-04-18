function [pks,locs] = getFreqs(filename,inicio,fim) 
  
sinalOut = 'sinalOut.wav';

%amostra o sinal e cria uma saida de audio em uma janela de tempo
samples = [inicio,fim];
[y,Fs] = audioread(filename,samples);
audiowrite(sinalOut,y,Fs);



[Y,f] = getFFTAudioFile(sinalOut);
%Y = Y/norm(Y);
%Peak = max(Y)*0.15;
%determinacao dos picos e frequencias presentes no audio

[pks,locs] = findpeaks(Y,f,'SortStr','descend');
%[pks,locs] = findpeaks(Y,f,'MinPeakDistance',1,'Npeaks',100,'MinPeakHeight',Peak,'SortStr','descend');
%[pks,locs] = findpeaks(Y,f,'MinPeakProminence',0.1,'Npeaks',100,'SortStr','descend');



end