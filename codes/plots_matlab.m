figure("NumberTitle", "off", "Name", "ReLU")

x = -5:0.1:5;
y = max(0.05*x, x);

plot(x, y);
xlabel("X");
ylabel("Y");
grid on;
axis on;
axis([-5, 5, -2, 5])
