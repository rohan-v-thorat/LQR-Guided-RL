clear
disp_rl_lqr = load('data/testing_response/disp_LQR_Guided_RL.mat').displacement;
disp_lqr = load('data/testing_response/disp_LQR.mat').displacement;

vel_rl_lqr = load('data/testing_response/vel_LQR_Guided_RL.mat').velocity;
vel_lqr = load('data/testing_response/vel_LQR.mat').velocity;

acc_rl_lqr = load('data/testing_response/acc_LQR_Guided_RL.mat').acceleration;
acc_lqr = load('data/testing_response/acc_LQR.mat').acceleration;

force_rl_lqr = load('data/testing_response/force_LQR_Guided_RL.mat').force;
force_lqr = load('data/testing_response/force_LQR.mat').force;

t = linspace(0,20,1000);

fig1 = figure('Color','white','Position',[400 80 1*1000 1*700]);
subplot(4,1,1)
plot(t,disp_lqr,'LineWidth',1.2,'Color','blue')
hold on
plot(t,disp_rl_lqr,'LineWidth',1.2,'Color','red')
ylabel('Disp (m)','FontSize',11)

subplot(4,1,2)
plot(t,vel_lqr,'LineWidth',1.2,'Color','blue')
hold on
plot(t,vel_rl_lqr,'LineWidth',1.2,'Color','red')
ylabel('Vel (m/s)','FontSize',11)

subplot(4,1,3)
plot(t,acc_lqr,'LineWidth',1.2,'Color','blue')
hold on
plot(t,acc_rl_lqr,'LineWidth',1.2,'Color','red')
ylabel('Acc (m/s^2)','FontSize',11)

subplot(4,1,4)
plot(t,force_lqr,'LineWidth',1.2,'Color','blue')
hold on
plot(t,force_rl_lqr,'LineWidth',1.2,'Color','red')
xlabel('Time (s)','FontSize',11)
ylabel('Control force (N)','FontSize',11)

legend({'LQR','LQR-Guided RL'},'Position',[0.47 0.905 0.1 0.12],'Orientation','horizontal','FontSize',10);

export_fig('plots\LQR_Guided_RL_testing',"-png")