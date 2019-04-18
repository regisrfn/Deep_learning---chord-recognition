function [Y,fx,Fs] = dft(filename)
% Computes Discrete Fourier Transform
% -----------------------------------
% [Xk] = dft(xn,N)
% Xk = DFT coeff. array over 0 <= k <= N-1
% xn = N-point finite-duration sequence
% N = Length of DFT
%

[y,Fs] = audioread(filename);
N = length(y); fx = Fs*(0:N/2-1)/N;


n = 0:1:N-1; % row vector for n
k = 0:1:N-1; % row vecor for k
WN = exp(-1i*2*pi/N); % Wn factor
nk = n'*k; % creates a N by N matrix of nk values
WNnk = WN .^ nk; % DFT matrix
Xk = y * WNnk; % row vector for DFT coefficients

Y = abs(Xk);


plot(Y)