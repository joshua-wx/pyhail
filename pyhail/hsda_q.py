Ac = phi / 600
Bc = 1.0 / ( 10 ** (0.1 * snr))
Cc = phi / 300
Dc = (1 - rhv) / 0.5
Ec = 3.16228 / (10 ^ (0.1 * SNR))
Fc = phi /100
Gc = Dc 
Hc = 1 / (10 ** (0.1 * snr))

if rhv < 0.8 and dbzh < 25:
    Dc = 0
    Gc = 0

block_threshold = 0.5
R = (precent_blocked)/(block_threshold) OR phi

Q_z   = exp( -0.69 * (Ac**2 + Bc**2 + T**2))
Q_zdr = exp( -0.69 * (Cc**2 + Dc**2 + T**2))
Q_rhv = exp( -0.69 * (Fc**2 + Gc**2 + Hc**2))