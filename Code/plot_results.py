import matplotlib.pyplot as plt
import numpy as np

# Plot and analyse results
def autolabel(ax, rects,i):
    """Attach a text label above each bar """
    for rect in rects:
        height = rect.get_height()
        ax[i].annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def plot(results):
    width= 0.2
    y_preds = [results[0][0], results[4][0], results[8][0], results[12][0], results[16][0]]
    y_preds1 = [results[1][0], results[5][0], results[9][0], results[13][0], results[17][0]]
    y_preds2 = [results[2][0], results[6][0], results[10][0], results[14][0], results[18][0]]
    y_preds3 = [results[3][0], results[7][0], results[11][0], results[15][0], results[19][0]]

    y_preds4 = [results[0][1], results[4][1], results[8][1], results[12][1], results[16][1]]
    y_preds5 = [results[1][1], results[5][1], results[9][1], results[13][1], results[17][1]]
    y_preds6 = [results[2][1], results[6][1], results[10][1], results[14][1], results[18][1]]
    y_preds7 = [results[3][1], results[7][1], results[11][1], results[15][1], results[19][1]]

    model_names = ['BASE','FCN_SINE','FCN_ELU','FCN_Residual', 'FCN_DCT']
    losses = ['huber_loss','mean_squared_error','mean_absolute_error','log_cosh']
    fig, ax = plt.subplots(2,figsize=(12,10))
    labels = model_names

    data = np.round(y_preds,3)
    data_1 = np.round(y_preds1,3)
    data_2 = np.round(y_preds2,3)
    data_3 = np.round(y_preds3,3)

    data_4 = np.round(y_preds4,3)
    data_5 = np.round(y_preds5,3)
    data_6 = np.round(y_preds6,3)
    data_7 = np.round(y_preds7,3)

    x = np.arange(len(labels))  # the label locations


    rects1 = ax[0].bar(x - (width+0.3)/2, data, width, label="huber")
    rects2 = ax[0].bar(x - (width+0.15)/4, data_1, width, label="mse")
    rects3 = ax[0].bar(x + (width+0.15)/4, data_2, width, label="mae")
    rects4 = ax[0].bar(x + (width+0.3)/2, data_3, width, label="log-cosh")

    rects5 = ax[1].bar(x - (width+0.3)/2, data_4, width, label="huber")
    rects6 = ax[1].bar(x - (width+0.15)/4, data_5, width, label="mse")
    rects7 = ax[1].bar(x + (width+0.15)/4, data_6, width, label="mae")
    rects8 = ax[1].bar(x + (width+0.3)/2, data_7, width, label="log-cosh")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss of each classifier (test)')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    ax[0].legend()

    ax[1].set_ylabel('MAE')
    ax[1].set_title('MAE of each classifier (test)')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    ax[1].legend()

    autolabel(ax,rects1,0)
    autolabel(ax,rects2,0)
    autolabel(ax,rects3,0)
    autolabel(ax,rects4,0)

    autolabel(ax,rects5,1)
    autolabel(ax,rects6,1)
    autolabel(ax,rects7,1)
    autolabel(ax,rects8,1)


    fig.tight_layout()

    plt.show()