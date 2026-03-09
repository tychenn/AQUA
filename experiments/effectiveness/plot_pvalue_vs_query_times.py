import matplotlib.pyplot as plt
import matplotlib.ticker 
import numpy as np

def plot_four_lines(lines_data):

    fig, ax = plt.subplots(figsize=(9, 9))
    markers = ['o', 'D', 's', 'P'] 
    linestyles = ['-', '-', '-', '-'] 
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plotted_line_count = 0

    for i, line_item in enumerate(lines_data):
        current_x_vals = line_item.get('watermark_num')
        current_y_vals = line_item.get('retrieval_ratio')


        x_vals = np.array(current_x_vals)
        y_vals = np.array(current_y_vals)
        name = line_item.get('name', f'Series {i+1}')
        marker_style = markers[i % len(markers)]
        linestyle_style = linestyles[i % len(linestyles)] 
        current_series_color = color_cycle[i % len(color_cycle)]
        if i == 1: 
            current_linewidth = 4.5
            current_markersize = 9
            current_alpha = 0.55
        else: 
            current_linewidth = 4.5
            current_markersize = 9
            current_alpha = 0.55

        ax.plot(x_vals, y_vals, label=name, marker=marker_style,
                linestyle=linestyle_style, color=current_series_color,
                linewidth=current_linewidth, markersize=current_markersize,
                alpha=current_alpha)
        plotted_line_count += 1
    ax.axhline(y=0.05, color='gray', linestyle='--', linewidth=4, label='$\\alpha$')
    ax.text(1.02, 0.98, '$\\alpha$', va='center', ha='left', fontsize=38, color='gray', transform=ax.transAxes)
    ax.set_xlabel("Query Times", fontsize=29) 
    ax.set_ylabel("p-value", fontsize=29) 
    ax.tick_params(axis='both', which='major', labelsize=23)
    ax.set_yscale('log')
    ax.set_xlim([0, 520])
    ax.set_ylim([1e-50, 1.5]) 
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()) 


    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    if plotted_line_count > 0 or True: 
         ax.legend(loc="best", fontsize=23) 
    ax.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    save_filename = "experiments/effectiveness/pvaluenum.png" 
    fig.savefig(save_filename, dpi=300)
    

if __name__ == '__main__':
    x_custom_data = np.array([20, 50, 100, 200, 300,500])
    num_points = len(x_custom_data) 
    # This is merely a drawing code. You need to get the pvalue in table.py
    lines_to_plot_example = [
        {
            'watermark_num': x_custom_data,
            'retrieval_ratio': [0.467,0.208,0.093,0.0088796,0.0063,0.00324658],
            'name': 'Naive'
        },
        {
            'watermark_num': x_custom_data,
            'retrieval_ratio': [0.68,0.36,0.12,0.0345,0.023,0.014],
            'name': 'Opt.'
        },
        {
            'watermark_num': x_custom_data,
            'retrieval_ratio': [0.061,0.000168,5.84e-11,1.1968e-19,2.45e-32,1.66e-47],
            'name': 'AQUA_acronym'
        },
        {
            'watermark_num': x_custom_data,
            'retrieval_ratio': [0.141,0.000968,8.84e-5,1.4968e-9,2.45e-18,1.66e-21],
            'name': 'AQUA_spatial'
        }
    ]
    plot_four_lines(lines_to_plot_example)
    print("-" * 30)
