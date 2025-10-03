clear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RL has -2 to 2 output range for no prior
% RL has -1 to 1 output range when LQR is prior
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = load('data\training_response\train_response_RL.mat').train_response;
data = reshape(data,999,8)';

data_lqr = load('data\training_response\train_response_LQR_Guided_RL.mat').train_response;
data_lqr = reshape(data_lqr,999,8)';

data_uncont = load('data\Uncontrolled\acc_uncont.mat').acceleration;

t = linspace(0,20,999);

fig1 = figure('Color','white','Position',[400 80 1*1000 1*300]);
plot(t,data(2,:),'LineWidth',1.2,'Color','blue')
hold on
plot(t,data_lqr(2,:),'LineWidth',1.2,'Color','red')
% legend({'RL','LQR-Guided RL'},'Position',[0.47 0.905 0.1 0.12],'Orientation','horizontal','FontSize',10);
legend({'RL','LQR-Guided RL'},'FontSize',10);
xlabel('Time (s)','FontSize',11)
ylabel('Acc (m/s^2)','FontSize',11)
export_fig('plots\LQR_Guided_RL_training',"-png")

fig2 = figure('Color','white','Position',[400 80 1*1000 1*300]);
plot(t,data(2,:),'LineWidth',1.2,'Color','blue')
hold on
plot(t,data_uncont(1,1:end-1),'LineWidth',1.2,'Color','red')
% legend({'RL','Uncontrolled'},'Position',[0.47 0.905 0.1 0.12],'Orientation','horizontal','FontSize',10);
legend({'RL','Uncontrolled'},'FontSize',10);
xlabel('Time (s)','FontSize',11)
ylabel('Acc (m/s^2)','FontSize',11)
export_fig('plots\RL_training',"-png")