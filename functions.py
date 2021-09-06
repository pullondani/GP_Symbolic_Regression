import math
import csv
# import matplotlib.pyplot as plt

## Function to create data for the GP to try and match
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
    with open('data.txt', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        for i, d in enumerate(inp):
            writer.writerow([d, out[i]])




def main():
    # Step size and number of data points
    inp = [x/100 for x in range(-500, 1000, 25)]
    out = f1(inp)

    writeTo(inp, out)
    print('Job done.')

    # plt.plot(inp, out, label='Function Output')
    # plt.legend()
    # plt.title('GP Function Output')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # plt.show()


if __name__ == '__main__':
    main()