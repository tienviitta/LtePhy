%# LTE OFDM Octave demo script
pkg load communications
rand("seed", 12345);
randn("seed", 12345);
clear all
%close all

%# Chest flag
LSCHEST = false;
MMSECHEST = true;
tau_rms = 1; % rms delay estimate???

%# Params
N_subf = 100;
%N_subf = 1;
%# 20 MHz system bandwidth (Note! Slow!)
%N_prbs = 100;
%N_fft = 2048;
%fs = 30.72e6;
%N_cp = 144;
%# 1.4 MHz system bandwidth
N_prbs = 6;
N_fft = 128;
fs = 1.920e6;
N_cp = 9;
%# Note! No need to change these!
M_data = 16; % 16QAM data symbols modulation order
N_subc = 12;
N_syms = 14;
N_gb = (N_fft - (N_prbs*N_subc))/2;

%# LTE grid
G_lte = zeros(N_prbs*N_subc, N_syms);

%# CRS syms (Note! Random QPSK!)
C_subc = 1:6:N_prbs*N_subc;
C_syms = [1 5 8 12]; %# Tx0
N_crs  = length(C_subc)*length(C_syms);
S_crs = pskmod(randi(4, length(C_subc), length(C_syms))-1, 4, pi/4, "gray");

%# Map CRS
for sym_i = 1:length(C_syms)
    G_lte(C_subc,C_syms(sym_i)) = S_crs(:,sym_i); 
end

%# Data symbols allocation
N_data = sum(sum(G_lte == 0));
S_map = find(G_lte == 0);

%# FD MMSE initialization
if MMSECHEST
Np = N_crs/length(C_syms);
Nps = 6;
df = 1/N_fft;  %1/(ts*Nfft);
j2pi_tau_df = j*2*pi*tau_rms*df;
K1 = repmat([0:N_prbs*N_subc-1].',1,Np);
K2 = repmat([0:Np-1],N_prbs*N_subc,1);
rf = 1./(1+j2pi_tau_df*(K1-K2*Nps));
K3 = repmat([0:Np-1].',1,Np);
K4 = repmat([0:Np-1],Np,1);
rf2 = 1./(1+j2pi_tau_df*Nps*(K3-K4));
Rhp = rf;
end

SNR = [0:2:20];
%SNR = 30;
bers = zeros(length(SNR),N_subf);
for SNR_i = 1:length(SNR)
  if MMSECHEST
  snr = 10^(SNR(SNR_i)*0.1);
  Rpp = rf2 + eye(Np)/snr;
  W_MMSE = Rhp*inv(Rpp);
  end
  for SF_i = 1:N_subf

    %# Data bits and modulation
    B_data = randi(M_data, 1, N_data)-1;
    S_data = sqrt(1/10)*qammod(B_data, M_data);

    %# Map data symbols
    G_lte(S_map) = S_data;

    %# Map OFDM symbols
    G_sf = [
        zeros(1, N_syms); ... % DC subcarrier
        G_lte(N_prbs*N_subc/2+1:end,:); ... % Positive subcarriers
        zeros(N_gb-1, N_syms); ... % Guard band
        zeros(N_gb, N_syms); ... % Guard band
        G_lte(1:N_prbs*N_subc/2,:); ... % Negative subcarriers
    ];

    %# OFDM syms (Note! Same CP for every OFDM symbol to save computation!)
    s_syms = ifft(G_sf, N_fft, 1);
    s_cp = s_syms(end-N_cp+1:end,:);
    tx_syms = [s_cp;s_syms];
    tx_sig = sqrt(N_fft)*tx_syms(:);

    %# Channel (Uncorrelated multi-path Rayleigh fading + AWGN)
    h = [
        0; ...
        (randn + 1i*randn)/2; ...
        (randn + 1i*randn); ...
        (randn + 1i*randn)/2; ...
        0; ...
        (randn + 1i*randn)/4 ...
    ];
    h = h./sqrt(sum(h)); %# Note! Normalize due to the SNR setting!
    ch_sig = conv(h, tx_sig);
    rx_sig = awgn(ch_sig, SNR(SNR_i)); %# AWGN

    %# OFDM syms
    rx_syms = reshape(rx_sig(1:N_fft*N_syms+N_cp*N_syms), N_cp+N_fft, N_syms);
    rx_syms = rx_syms(N_cp+1:end,:);
    R_sf = fftshift(fft(rx_syms, N_fft, 1),1);

    %# De-Map CRS and data symbols
    Y_sf = [ ...
        R_sf(N_gb+1:N_gb+N_prbs*N_subc/2,:); ...
        R_sf(N_gb+N_prbs*N_subc/2+2:N_gb+1+N_prbs*N_subc,:) ...
    ];
    
    %# De-Map CRS and scale
    for sym_i = 1:length(C_syms)
        R_crs(:,sym_i) = sqrt(1/N_fft)*Y_sf(C_subc,C_syms(sym_i)); 
    end
    
    %# Chest (LS with FD and TD linear interpolation)
    if LSCHEST
    H_LS = R_crs .* conj(S_crs ); %# LS channel estimates
    H_fd = interp1(C_subc.', H_LS, (1:N_prbs*N_subc).', 'linear', 'extrap');
    H_td = interp1(C_syms, H_fd.', (1:N_syms), 'linear', 'extrap').';
    end
    
    %# Chest (MMSE FD and TD linear interpolation)
    if MMSECHEST
    H_LS = R_crs .* conj(S_crs ); %# LS channel estimates
    H_fd = W_MMSE*H_LS;  %# FD MMSE channel estimate
    H_td = interp1(C_syms, H_fd.', (1:N_syms), 'linear', 'extrap').';
    end

    %# De-Map data
    Y_data = Y_sf(S_map);
    
    %# Equalizer
    if LSCHEST || MMSECHEST
    H_data = H_td(S_map);
    Y_eq = (Y_data .* conj(H_data))./(abs(H_data).^2);
    else
    Y_eq = Y_data;
    end
    
    % Demod
    R_data = qamdemod(sqrt(10/N_fft)*Y_eq, M_data);

    %# Results
    [berr, ber] = biterr(B_data.', R_data);
    bers(SNR_i,SF_i) = ber;
  
  end
  fprintf('.');
end
fprintf('\n');

disp('BERs:')
disp(mean(bers,2))
figure, semilogy(SNR, mean(bers,2), 'o-'), grid on
xlabel('SNR')
ylabel('BER')

%# Plotting (Last PSD estimate and channel estimate at all REs)
%pwelch(rx_sig,[],[],[],fs,'centerdc',[],[],[])
figure, mesh(1:N_syms, 1:N_prbs*N_subc, abs(H_td).^2)
xlabel('OFDM sym')
ylabel('subc')
zlabel('|H|^2')
