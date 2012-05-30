close all
clear all

data = load('profiling.dat');

ncards     = 1;
gflops_vec = 2;
bwidth_vec = 3;
gflops_rdc = 4;
bwidth_rdc = 5;
gflops_spm = 6;
bwidth_spm = 7;


vec_gflops = [];
vec_bwidth = [];
rdc_gflops = [];
rdc_bwidth = [];
spm_gflops = [];
spm_bwidth = [];

max_cards = max(data(:,ncards));

for n = 1:max_cards
    idx = find(data(:,ncards) == n);
    avg = sum(data(idx,:)) / length(idx);

    vec_gflops = [vec_gflops avg(gflops_vec)];
    vec_bwidth = [vec_bwidth avg(bwidth_vec)];
    rdc_gflops = [rdc_gflops avg(gflops_rdc)];
    rdc_bwidth = [rdc_bwidth avg(bwidth_rdc)];
    spm_gflops = [spm_gflops avg(gflops_spm)];
    spm_bwidth = [spm_bwidth avg(bwidth_spm)];
end

figure(1);
set(gca, 'FontSize', 18);

plot(1:max_cards, vec_gflops, 'ko-', ...
		'linewidth', 2, 'markersize', 6, 'markerfacecolor', 'w');
hold on
plot(1:max_cards, rdc_gflops, 'ro-', ...
		'linewidth', 2, 'markersize', 6, 'markerfacecolor', 'w');
hold on
plot(1:max_cards, spm_gflops, 'bo-', ...
		'linewidth', 2, 'markersize', 6, 'markerfacecolor', 'w');

xlim([0.75 max_cards + 0.25]);
set(gca, 'xtick', [1 2 3]);

xlabel('Number of devices (Tesla C2070)')
ylabel('Effective GFLOPS')

legend('Vector arithmetic', 'Reduction', 'SpMV', ...
		'location', 'northwest');
legend boxoff

print('-depsc', 'gflops.eps');

figure(2)

set(gca, 'FontSize', 18);

plot(1:max_cards, vec_bwidth, 'ko-', ...
		'linewidth', 2, 'markersize', 6, 'markerfacecolor', 'w');
hold on
plot(1:max_cards, rdc_bwidth, 'ro-', ...
		'linewidth', 2, 'markersize', 6, 'markerfacecolor', 'w');
hold on
plot(1:max_cards, spm_bwidth, 'bo-', ...
		'linewidth', 2, 'markersize', 6, 'markerfacecolor', 'w');

xlim([0.75 max_cards + 0.25]);
set(gca, 'xtick', [1 2 3]);

xlabel('Number of devices (Tesla C2070)')
ylabel('Effective bandwidth (GB/sec)')

legend('Vector arithmetic', 'Reduction', 'SpMV', ...
		'location', 'northwest');
legend boxoff

print('-depsc', 'bwidth.eps');

