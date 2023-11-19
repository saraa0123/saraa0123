M = 16;
num_symbol = 1000
data = randi([0 M-1], num_symbol,1);
txSig = pskmod(data,M,pi/M);
scatterplot(txSig)
title('PSK Scatter Plot')


rxSig = awgn(txSig,20);
scatterplot(rxSig)
title('Noisy PSK Scatter Plot')

snr = []
BER_sim = []
ber_theoretical = []
for i = 1:1:50
	snr(i) = i
	EbNo = 10^(i/10);
	rxSig = awgn(txSig, EbNo);
	BER_sim(i) = (biterr(int16(abs(txSig)), int16(abs(rxSig))))/num_symbol;
	ber_theoretical(i) = erfc(sqrt(EbNo*((sin(pi/M))^2)))
end
 
plot(snr, BER_sim)
plot(snr, ber_theoretical)
legend('BER_sim ', ber_theoretical ')




