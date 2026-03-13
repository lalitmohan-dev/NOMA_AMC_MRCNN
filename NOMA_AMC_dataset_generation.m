% author : Ashok Parmar (fixed & scaled by Priyanshu & Lalit)
clc;
clear variables;
close all;

% ─────────────────────────────────────────
%  CONFIG — change these to scale dataset
% ─────────────────────────────────────────
N              = 200;       % signal length
SAMPLES_PER_SNR= 40000;      % ← was 1000, now 5000 → 80,000 total samples
Pf             = 0.8;
Pn             = 0.4;       % ratio = Pf/Pn = 2
vriance        = 0.1;       % Rayleigh fading variance
OUTPUT_FILE    = 'myfile_ratio2_large.mat';

% ─────────────────────────────────────────
%  How dataset size scales:
%
%  SAMPLES_PER_SNR = 1000  →  16,000  total  (current, overfitting)
%  SAMPLES_PER_SNR = 5000  →  80,000  total  (recommended)
%  SAMPLES_PER_SNR = 10000 → 160,000  total  (large, slow)
%  SAMPLES_PER_SNR = 50000 → 800,000  total  (original paper size)
% ─────────────────────────────────────────

% symbol sets
symbol_bpsk  = [-1+0i, 1+0i];
theta1       = cos(pi/8)+1i*sin(pi/8);
symbol_qpsk  = theta1 * [0.7071+0.7071i, -0.7071+0.7071i, ...
                          0.7071-0.7071i, -0.7071-0.7071i];
symbol_8psk  = [1, 0.7071+0.7071i, 1i, -0.7071+0.7071i, ...
               -1, -0.7071-0.7071i, -1i, 0.7071-0.7071i];
qam = [-1.0000 - 1.0000i
        1.0000 - 1.0000i
        1.0000 + 1.0000i
       -1.0000 + 1.0000i
       -3.0000 - 1.0000i
       -1.0000 - 3.0000i
        1.0000 - 3.0000i
        3.0000 - 1.0000i
        3.0000 + 1.0000i
        1.0000 + 3.0000i
       -1.0000 + 3.0000i
       -3.0000 + 1.0000i
       -3.0000 - 3.0000i
        3.0000 - 3.0000i
        3.0000 + 3.0000i
       -3.0000 + 3.0000i];
qam16        = transpose(qam);
sqrt_mean    = sqrt(mean(abs(qam16).^2));
symbol_16qam = qam16 / sqrt_mean;

setof_symbs = {symbol_bpsk, symbol_qpsk, symbol_8psk, symbol_16qam};

% pre-allocate for one SNR level at a time
true_lbls = zeros(1, SAMPLES_PER_SNR);
data_y    = zeros(SAMPLES_PER_SNR, N);
all_snrs  = zeros(1, SAMPLES_PER_SNR);
all_h     = zeros(1, SAMPLES_PER_SNR);
snr_itr   = zeros(1, SAMPLES_PER_SNR);

% output accumulators
snrs      = [];
data_Y    = [];
true_Mods = [];

% ─────────────────────────────────────────
%  Progress tracking
% ─────────────────────────────────────────
SNR_LEVELS   = -10:2:20;
total_iters  = length(SNR_LEVELS) * SAMPLES_PER_SNR;
completed    = 0;
t_start      = tic;

w = waitbar(0, 'Starting...');

fprintf('\n================================================\n');
fprintf('  NOMA Dataset Generation\n');
fprintf('================================================\n');
fprintf('  N              : %d symbols\n', N);
fprintf('  Samples/SNR    : %d\n', SAMPLES_PER_SNR);
fprintf('  SNR levels     : %d  (-10 to +20 dB)\n', length(SNR_LEVELS));
fprintf('  Total samples  : %d\n', total_iters);
fprintf('  Power ratio    : %.1f (Pf=%.1f, Pn=%.1f)\n', Pf/Pn, Pf, Pn);
fprintf('  Output file    : %s\n', OUTPUT_FILE);
fprintf('================================================\n\n');

% ─────────────────────────────────────────
%  Main generation loop
% ─────────────────────────────────────────
for snr_db = SNR_LEVELS
    snr        = 10^(snr_db/10);
    t_snr      = tic;

    for iter = 1:SAMPLES_PER_SNR
        % Rayleigh fading coefficient
        xr = randn(1);
        yi = randn(1);
        h  = sqrt(vriance * (xr^2 + yi^2));
        all_h(iter) = h;

        % random modulation for far and near user
        i = randi(4);
        j = randi(4);
        true_lbls(iter) = i - 1;   % 0-indexed label

        % sample symbols
        symb_far  = setof_symbs{i};
        symb_near = setof_symbs{j};
        xf = randsample(symb_far,  N, true);
        xn = randsample(symb_near, N, true);

        % NOMA mixing
        y0 = h * (sqrt(Pf)*xf + sqrt(Pn)*xn);
        ps = sum(abs(y0).^2) / numel(y0);

        % AWGN noise
        noisepower  = ps / snr;
        z           = sqrt(noisepower*0.5) * ...
                      complex(randn(size(y0)), randn(size(y0)));
        pn          = sum(abs(z).^2) / numel(z);

        % received signal
        y               = y0 + z;
        all_snrs(iter)  = ps / pn;
        data_y(iter, :) = y;
        snr_itr(iter)   = snr_db;

        % update waitbar every 100 samples (not every sample — faster)
        completed = completed + 1;
        if mod(iter, 100) == 0
            elapsed  = toc(t_start);
            eta      = (elapsed / completed) * (total_iters - completed);
            waitbar(completed / total_iters, w, ...
                sprintf('SNR=%+ddB | %d/%d | %.1f%% | ETA: %ds', ...
                snr_db, iter, SAMPLES_PER_SNR, ...
                completed/total_iters*100, round(eta)));
        end
    end

    % append this SNR block
    snrs      = [snrs      snr_itr  ];
    data_Y    = [data_Y;   data_y   ];
    true_Mods = [true_Mods true_lbls];

    t_snr_end = toc(t_snr);
    fprintf('  SNR %+4ddB  done  |  %5d samples  |  %.1fs\n', ...
            snr_db, SAMPLES_PER_SNR, t_snr_end);
end

close(w);

% ─────────────────────────────────────────
%  Save
% ─────────────────────────────────────────
fprintf('\n  Saving to %s ...\n', OUTPUT_FILE);
save(OUTPUT_FILE, 'data_Y', 'true_Mods', 'snrs', '-v7.3');

t_total = toc(t_start);
fprintf('\n================================================\n');
fprintf('  Dataset saved successfully!\n');
fprintf('================================================\n');
fprintf('  File           : %s\n', OUTPUT_FILE);
fprintf('  Total samples  : %d\n', size(data_Y, 1));
fprintf('  Signal length  : %d\n', size(data_Y, 2));
fprintf('  Labels range   : %d to %d\n', min(true_Mods), max(true_Mods));
fprintf('  SNR range      : %d to %d dB\n', min(snrs), max(snrs));
fprintf('  Time taken     : %.1f seconds (%.1f min)\n', ...
        t_total, t_total/60);
fprintf('================================================\n\n');