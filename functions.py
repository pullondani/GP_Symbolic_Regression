import math
import csv
import matplotlib.pyplot as plt

def f1(inp):
    out = []
    for x in inp:
        if x > 0:
            ans = (1/x) + math.sin(x)
            out.append(ans)
        else:
            ans = 2 * x + (x*x) + 3.0
            out.append(ans)
    return out


def writeTo(inp, out):
    with open('data.txt', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        for i, d in enumerate(inp):
            writer.writerow([d, out[i]])




def main():
    inp = [x/100 for x in range(0, 1000, 25)]
    out = f1(inp)

    writeTo(inp, out)

    # plt.plot(inp, out, label='Function Output')
    # plt.legend()
    # plt.title('GP Function Output')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.show()


if __name__ == '__main__':
    main()