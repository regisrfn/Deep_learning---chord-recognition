function [Y,f,Fs] = getFFTAudioFile(input_args)

[y,Fs] = audioread(input_args);
t=(0:length(y)-1)/Fs; %t = vetor de tempo ,onde Fs a taxa de amostragem e length(y) o numero de amostras
%sound(y,Fs);
N = length(t);
%flim = [0 2000];

%transformada rapida de fourier
NFFT = 2^nextpow2(N);
Y = abs(fft(y,NFFT)/NFFT);
Y = 2 * Y(1:NFFT/2+1);
f = Fs/2*linspace(0,1,NFFT/2+1);

% %Exibicao dos graficos
% figure;
% subplot(211); plot(t,y);
% xlabel('Tempo (s)')
% ylabel('Amplitude')
% title('Sinal de Audio'); grid on
% subplot(212); plot(f,Y);xlim(flim)
% xlabel('Frequencia (Hz)')
% ylabel('Amplitude')
% title('FFT'); grid on


end

