% author : Ashok Parmar (fixed by Priyanshu & Lalit)
clc;
clear variables;
close all;

N = 200;

% symbol sets
symbol_bpsk  = [-1+0i, 1+0i];
theta1       = cos(pi/8)+1i*sin(pi/8);
% theta2 removed — was defined but never used
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

% Power allocation (ratio = 4 as per your experiment)
Pf = 0.8; Pn = 0.2;

% Rayleigh fading variance
vriance = 0.1;

setof_symbs = {symbol_bpsk, symbol_qpsk, symbol_8psk, symbol_16qam};

% Pre-allocate correctly for 1000 samples
true_lbls = zeros(1, 1000);
data_y    = zeros(1000, N);
all_snrs  = zeros(1, 1000);
all_h     = zeros(1, 1000);
Mf_lik    = zeros(4, 16000);

snrs      = [];
data_Y    = [];
true_Mods = [];

% ✅ FIX 1 — proper waitbar with total progress tracking
total_iters = 16 * 1000;
completed   = 0;
w = waitbar(0, 'Generating NOMA dataset...');

for snr_db = -10:2:20
    snr = 10^(snr_db/10);

    for iter = 1:1000
        xr = randn(1);
        yi = randn(1);
        h  = sqrt(vriance * (xr^2 + yi^2));    % Rayleigh fading
        all_h(iter) = h;

        i = randi(4);
        true_lbls(iter) = i - 1;
        j = randi(4);

        % Select symbols for far and near user
        symb_far  = setof_symbs{i};
        symb_near = setof_symbs{j};

        xf = randsample(symb_far,  N, true);
        xn = randsample(symb_near, N, true);

        % NOMA signal mixing
        y0 = h * (sqrt(Pf)*xf + sqrt(Pn)*xn);
        ps = sum(abs(y0).^2) / numel(y0);

        % Add AWGN noise
        noisepower = ps / snr;
        z  = sqrt(noisepower*0.5) * complex(randn(size(y0)), randn(size(y0)));
        pn = sum(abs(z).^2) / numel(z);

        y      = y0 + z;
        snr_y  = ps / pn;

        all_snrs(iter)  = snr_y;
        data_y(iter, :) = y;
        snr_itr(iter)   = snr_db;

        % ✅ FIX 1 — update waitbar with TOTAL progress + info text
        completed = completed + 1;
        waitbar(completed / total_iters, w, ...
            sprintf('SNR = %d dB | Sample %d/1000 | Overall: %.1f%%', ...
            snr_db, iter, (completed/total_iters)*100));
    end

    snrs      = [snrs      snr_itr  ];
    data_Y    = [data_Y;   data_y   ];
    true_Mods = [true_Mods true_lbls];
end

close(w);

% ✅ FIX 2 — single clean save, removed duplicate old save line
save('myfile.mat', 'data_Y', 'true_Mods', 'snrs', '-v7.3');

disp('Dataset saved to myfile.mat');
disp(['Total samples : ' num2str(size(data_Y,1))]);
disp(['Signal length : ' num2str(size(data_Y,2))]);