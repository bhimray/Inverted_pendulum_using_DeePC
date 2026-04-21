M = 0.5;
m = 0.2;
b = 0.1;
I = 0.006;
g = 9.8;
l = 0.3;

p = I*(M+m)+M*m*l^2; %denominator for the A and B matrices

A = [0      1              0           0;
     0 -(I+m*l^2)*b/p  (m^2*g*l^2)/p   0;
     0      0              0           1;
     0 -(m*l*b)/p       m*g*l*(M+m)/p  0];
B = [     0;
     (I+m*l^2)/p;
          0;
        m*l/p];
C = [1 0 0 0;
     0 0 1 0];
D = [0;
     0];

states = {'x' 'x_dot' 'phi' 'phi_dot'};
inputs = {'u'};
outputs = {'x'; 'phi'};

sys_ss = ss(A,B,C,D,'statename',states,'inputname',inputs,'outputname',outputs);

poles = eig(A)

co = ctrb(sys_ss);
controllability = rank(co)

Q = C'*C;
Q(1,1) = 5000;
Q(3,3) = 100;
R = 1;
K = lqr(A,B,Q,R)

Ac = (A-B*K);
Bc = B;
Cc = C;
Dc = D;

states = {'x' 'x_dot' 'phi' 'phi_dot'};
inputs = {'r'};
outputs = {'x'; 'phi'};

sys_cl = ss(Ac,Bc,Cc,Dc,'statename',states,'inputname',inputs,'outputname',outputs);

t = 0:0.01:5;
r =0.2*ones(size(t));

[y,t,x]=lsim(sys_cl,r,t);
u = r' - (K*x')';

figure
subplot(3,1,1)
plot(t,y(:,1),'LineWidth',2)
ylabel('cart position (m)')
title('LQR Response')
grid on

subplot(3,1,2)
plot(t,y(:,2),'LineWidth',2)
ylabel('pendulum angle (rad)')
grid on

subplot(3,1,3)
plot(t,u,'LineWidth',2)
xlabel('Time (s)')
ylabel('control input u')
grid on
