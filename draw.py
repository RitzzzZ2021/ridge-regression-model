import matplotlib.pyplot as plt

x = list(range(50))
file1 = open("gradient_loss", "r")
file2 = open("conjucate_loss", "r")
file3 = open("quasiNewton_loss", "r")
y_1 = file1.read().split(" ")
y_2 = file2.read().split(" ")
y_3 = file3.read().split(" ")
y1 = [float(i) for i in y_1]
y2 = [float(i) for i in y_2]
y3 = [float(i) for i in y_3]

plt.plot(x, y1)
plt.plot(x, y2, color='red', linestyle='--')
plt.plot(x, y3, color='green', linestyle=':')
plt.show()
