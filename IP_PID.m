M = 0.5;
m = 0.2;
b = 0.1;
I = 0.006;
g = 9.8;
l = 0.3;
q = (M+m)*(I+m*l^2)-(m*l)^2;
s = tf('s');
P_pend = (m*l*s/q)/(s^3 + (b*(I + m*l^2))*s^2/q - ((M + m)*m*g*l)*s/q - b*m*g*l/q);
P_cart = (((I+m*l^2)/q)*s^2 - (m*g*l/q))/(s^4 + (b*(I + m*l^2))*s^3/q - ((M + m)*m*g*l)*s^2/q - b*m*g*l*s/q);

Kp = 100;
Ki = 1;
Kd = 20;
C = pid(Kp,Ki,Kd);
T = feedback(P_pend,C);
T2 = feedback(1,P_pend*C)*P_cart;

%% Pendulum and cart response given input disturbance
t=0:0.01:10;
[phi_resp, t_phi] = impulse(T,t);
[x_resp, t_x] = impulse(T2,t);

figure
subplot(2,1,1)
plot(t_phi,phi_resp,'LineWidth',2)
title('Pendulum Angle Response (PID)')
xlabel('Time (s)')
ylabel('\phi (rad)')
grid on

subplot(2,1,2)
plot(t_x,x_resp,'LineWidth',2)
title('Cart Position Response (PID)')
xlabel('Time (s)')
ylabel('x (m)')
grid on
