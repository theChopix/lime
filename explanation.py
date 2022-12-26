import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns


class Explanation:

    def __init__(self, data, plot_title=None, x_label=None):
        # self.data = list(data.items())
        self.data = data
        self.plot_title = plot_title
        # self.subtitle = subtitle
        self.x_label = x_label

    def text_plot(self):
        df = pd.DataFrame(self.data, columns=["Features", "Values"])
        df.set_index("Features", inplace=True)
        return df

    def graph_plot(self, gradient=False):
        # self.data.reverse()

        dt = list(reversed(self.data))
        df = pd.DataFrame(dt, columns=["Features", "Values"])
        df.set_index("Features", inplace=True)

        index = df.index
        values = df['Values']
        plot_title = self.plot_title
        title_size = 18
        subtitle = "" # remove
        x_label = self.x_label

        fig, ax = plt.subplots(figsize=(8,5), facecolor=(.94, .94, .94))

        bar = ax.barh(index, values)
        plt.tight_layout()
        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))

        title = plt.title(plot_title, pad=20, fontsize=title_size)
        title.set_position([.33, 1])
        plt.subplots_adjust(top=0.9, bottom=0.1)

        ax.grid(zorder=0)

        if gradient:
            def gradientbars(bars):
                grad = np.atleast_2d(np.linspace(0,1,256))
                # grad = np.atleast_2d(np.linspace(100,1,200))
                ax_ = bars[0].axes
                lim = ax_.get_xlim()+ax_.get_ylim()
                for bar_ in bars:
                    bar_.set_zorder(1)
                    bar_.set_facecolor('none')
                    x,y = bar_.get_xy()
                    w, h = bar_.get_width(), bar_.get_height()
                    ax_.imshow(grad, extent=[x+w, x, y, y+h], aspect='auto', zorder=1)
                ax_.axis(lim)

            gradientbars(bar)

        # Set labels
        rects = ax.patches
        # For each bar: place a label
        for rect in rects:
            # Get X and Y placement of label from rect
            x_value = rect.get_width()
            y_value = rect.get_y() + rect.get_height() / 2

            # Number of points between bar and label; change to your liking
            space = -40
            # Vertical alignment for positive values
            ha = 'left'

            # If value of bar is negative: place label left of bar
            if x_value < 0:
                # Invert space to place label to the left
                space *= -1
                # Horizontally align label at right
                ha = 'right'

            # Use X value as label and format number with one decimal place
            label = '{:,.4f}'.format(x_value)

            # Create annotation
            plt.annotate(
                label,  # Use `label` as label
                (x_value, y_value),  # Place label at end of the bar
                xytext=(space, 0),  # Horizontally shift label by `space`
                textcoords='offset points',  # Interpret `xytext` as offset in points
                va='center',  # Vertically center label
                ha=ha,  # Horizontally align label differently for positive and negative values
                color='white')  # Change label color to white

        # Set subtitle
        trans = ax.get_xaxis_transform()  # x in data untis, y in axes fraction
        ann = ax.annotate(subtitle, xy=(5, 1), xycoords=trans,
                          bbox=dict(boxstyle='square,pad=1.3', fc='#f0f0f0', ec='none'))

        # Set x-label
        ax.set_xlabel(x_label, color='#525252')

        return mpl.pyplot.viridis()

    


