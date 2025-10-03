clear
disp_uncont = load('data/Uncontrolled/disp_uncont.mat').displacement;
disp_lqr = load('data/testing_response/disp_LQR.mat').displacement;

vel_uncont = load('data/Uncontrolled/vel_uncont.mat').velocity;
vel_lqr = load('data/testing_response/vel_LQR.mat').velocity;

acc_uncont = load('data/Uncontrolled/acc_uncont.mat').acceleration;
acc_lqr = load('data/testing_response/acc_LQR.mat').acceleration;

t = linspace(0,20,1000);

fig1 = figure('Color','white','Position',[400 80 1*1000 1*500]);
subplot(3,1,1)
plot(t,disp_uncont,'LineWidth',1.2,'Color','blue')
hold on
plot(t,disp_lqr,'LineWidth',1.2,'Color','red')
ylabel('Disp (m)','FontSize',11)

subplot(3,1,2)
plot(t,vel_uncont,'LineWidth',1.2,'Color','blue')
hold on
plot(t,vel_lqr,'LineWidth',1.2,'Color','red')
ylabel('Vel (m/s)','FontSize',11)

subplot(3,1,3)
plot(t,acc_uncont,'LineWidth',1.2,'Color','blue')
hold on
plot(t,acc_lqr,'LineWidth',1.2,'Color','red')
xlabel('Time (s)','FontSize',11)
ylabel('Acc (m/s^2)','FontSize',11)

legend({'Uncontrolled','LQR'},'Position',[0.47 0.905 0.1 0.12],'Orientation','horizontal','FontSize',10);

export_fig('plots\LQR_vs_Uncontrolled',"-png")