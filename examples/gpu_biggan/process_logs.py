import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("Parser for experiment logs")
parser.add_argument('--data_dir', required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    G_loss = []
    D_loss_real = []
    D_loss_fake = []
    steps = []

    with open(args.data_dir + 'G_loss.log') as f:
        for line in f.readlines():
            G_loss.append(float(line.split(':')[1]))
            steps.append(int(line.split(':')[0]))

    with open(args.data_dir + 'D_loss_real.log') as f:
        for line in f.readlines():
            D_loss_real.append(float(line.split(':')[1]))


    with open(args.data_dir + 'D_loss_fake.log') as f:
        for line in f.readlines():
            D_loss_fake.append(float(line.split(':')[1]))

    # Keep one every 8 elements (TPU cores)
    G_loss = G_loss[::8]
    D_loss_real = D_loss_real[::8]
    D_loss_fake = D_loss_fake[::8]
    steps = steps[::8]

    plt.figure()
    plt.plot(steps, G_loss)
    plt.plot(steps, D_loss_real)
    plt.plot(steps, D_loss_fake)
    plt.legend(['G_loss', 'D_loss_real', 'D_loss_fake'])
    plt.show()
