clear;
close all; 
clc;

% Solve y = Theta * s for "s"
n = 1000;  % dimension of s
p = 200;   % number of measurements, dim(y)
Theta = randn(p,n);  
y = randn(p,1);

% L1 minimum norm solution s_L1
cvx_begin;
   variable s_L1(n); 
   minimize( norm(s_L1,1) );    %min||S_1|| s.t. y=Theta*s
   subject to  
        Theta*s_L1==y;  
cvx_end;

% L2 minimum norm solution s_L2
s_L2 = pinv(Theta)*y; %pinv is to find psuedo inverse of Theta


%%
figure
subplot(3,2,1)
plot(s_L1,'b','LineWidth',1.5)
title('L1 Solution','Color','k','FontSize',24);
ylim([-.2 .2]), grid on

subplot(3,2,2)
plot(s_L2,'r','LineWidth',1.5)
title('L2 Solution','Color','k','FontSize',24);
ylim([-.2 .2]), grid on

subplot(3,2,[3 5])
[hc,h] = hist(s_L1,[-.1:.01:.1]);
bar(h,hc,'b')
axis([-.1 .1 -50 1000])

subplot(3,2,[4 6])
[hc,h] = hist(s_L2,[-.1:.01:.1]);
bar(h,hc,'r')
axis([-.1 .1 -20 400])

