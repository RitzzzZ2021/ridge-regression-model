import matplotlib.pyplot as plt

iteration = 100
x = list(range(iteration))
file1 = open("gradient_loss", "r")
file2 = open("conjucate_loss", "r")
file3 = open("quasiNewton_loss", "r")
y_1 = file1.read().split(" ")
y_2 = file2.read().split(" ")
y_3 = file3.read().split(" ")
y1 = [float(i) for i in y_1]
y2 = [float(i) for i in y_2]
y3 = [float(i) for i in y_3]


plt.plot(x, y1, label = 'gradient descent')
plt.plot(x, y2, label = 'conjugate descent', color='red', linestyle='--')
plt.plot(x, y3, label = 'quasi-Newton method', color='green', linestyle=':')
# plt.title('abalone')
# plt.title('bodyfat')
plt.title('housing')
plt.xlabel('iteration')
plt.ylabel('Mean Square Error')
plt.legend()
plt.savefig("result.png")
