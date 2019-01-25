import numpy as np
import matplotlib.pyplot as plt
from src.data_utils import read_class_definition

def plot_image(ax, x, y=None):
    if not y is None:
        print('Class: %s' % y)

    ax.imshow(x, cmap='gray')
    ax.axis('off')


def plot_array(fig, X, Y, classes_to_plot=None, samples_per_class=7):
    if classes_to_plot is None:
        classes_to_plot = np.unique(Y)
    num_classes = len(classes_to_plot)

    for k, y in enumerate(classes_to_plot):
        idxs = np.flatnonzero(Y == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        #print(y, idxs)

        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + k + 1
            ax = fig.add_subplot(samples_per_class, num_classes, plt_idx)
            ax.imshow(X[idx], cmap='gray')
            ax.axis('off')

def plot_confusion_matrix(
    ax, matrix, labels, title='Confusion matrix', fontsize=9):

    ax.set_xticks([x for x in range(len(labels))])
    ax.set_yticks([y for y in range(len(labels))])
    # Place labels on minor ticks
    ax.set_xticks([x + 0.5 for x in range(len(labels))], minor=True)
    ax.set_xticklabels(labels, rotation='90', fontsize=fontsize, minor=True)
    ax.set_yticks([y + 0.5 for y in range(len(labels))], minor=True)
    ax.set_yticklabels(labels[::-1], fontsize=fontsize, minor=True)
    # Hide major tick labels
    ax.tick_params(which='major', labelbottom='off', labelleft='off')
    # Finally, hide minor tick marks
    ax.tick_params(which='minor', width=0)

    # Plot heat map
    proportions = [1. * row / sum(row) for row in matrix]
    ax.pcolor(np.array(proportions[::-1]), cmap=plt.cm.Reds)

    # Plot counts as text
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            confusion = matrix[::-1][row][col]
            if confusion != 0:
                ax.text(col + 0.5, row + 0.5, confusion, fontsize=fontsize,
                    horizontalalignment='center',
                    verticalalignment='center')

    # Add finishing touches
    ax.grid(True, linestyle=':')
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel('prediction', fontsize=fontsize)
    ax.set_ylabel('actual', fontsize=fontsize)
    # fig.tight_layout()
    plt.show()

def make_scatterplot(X, y, feature1, feature2, class_names=None, class_definition=None):
    if class_names is None:
        class_names = np.unique(y)
    if class_definition is None:
        class_definition = read_class_definition()

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_title('Scatter plot: %s vs. %s' % (feature2, feature1))

    for class_name in class_names:
        class_label = class_definition['name_to_label'][class_name]
        class_color = class_definition['colors'][class_label]
        ax.scatter(X[feature1][y==class_label],
                   X[feature2][y==class_label],
                   c=class_color, 
                   s=15)
    ax.legend(class_names)
    ax.grid()


