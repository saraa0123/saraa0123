M = 16;
num_symbol = 1000
data = randi([0 M-1], num_symbol,1);

txSig = qammod (data, M);
scatterplot(txSig)
title('QAM Scatter Plot')
 

rxSig = awgn(txSig,20);
scatterplot(rxSig)
title('Noisy QAM Scatter Plot')

snr = []
BER_sim = []
ber_theoretical = []
for i = 1:1:50
	snr(i) = i
	EbNo = 10^(i/10);
	rxSig = awgn(txSig, EbNo);
	BER_sim(i) = (biterr(int16(abs(txSig)), int16(abs(rxSig))))/num_symbol;
	ber_theoretical(i) = 2*erfc(sqrt(0.4*EbNo))
end
 
plot(snr, BER_sim)
plot(snr, ber_theoretical)
legend('BER_sim ', ber_theoretical ')





