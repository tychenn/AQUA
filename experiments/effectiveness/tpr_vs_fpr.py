import matplotlib.pyplot as plt
import numpy as np

def _plot_roc_curve_core(fpr, tpr, title_suffix, save_filename, 
                         ax=None, curve_label=None, plot_random_guess=True,
                         marker='o', point_size=50, series_color=None): 
    create_new_figure = ax is None
    current_ax = ax

    if create_new_figure:
        fig, current_ax = plt.subplots(figsize=(8, 8)) 
    else:
        fig = current_ax.get_figure()

    label_text = curve_label if curve_label else ('' if create_new_figure else None)
    edge_c = series_color
    if series_color is None and create_new_figure: 
        edge_c = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    elif series_color is None and not create_new_figure: 
        edge_c = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]


    current_ax.scatter(fpr, tpr, label=label_text, s=point_size, marker=marker, 
                       facecolors='none', edgecolors=edge_c,linewidth=1.5) 

    if create_new_figure or plot_random_guess:
        random_guess_label = 'random' if create_new_figure else None
        current_ax.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label=random_guess_label)

    if create_new_figure:
        current_ax.set_xlim([0.0, 1.0])
        current_ax.set_ylim([0.0, 1.0]) 
        current_ax.set_xlabel(fontsize=12)
        current_ax.set_ylabel(fontsize=12)
        if title_suffix:
            plot_title += f" - {title_suffix}"
        current_ax.set_title(plot_title, fontsize=14)
        current_ax.tick_params(axis='both', which='major', labelsize=12)
        if label_text or random_guess_label:
            current_ax.legend(loc="lower right", fontsize=10)
        current_ax.grid(True, linestyle=':', alpha=0.9)
        current_ax.set_aspect('equal', adjustable='box')


def plot_multiple_roc_curves_from_fpr_tpr(curves_data, title="FPR vs. TPR of 4 Watermark Methods"):
    fig, ax = plt.subplots(figsize=(10, 8)) 
    ax.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label='Random')
    markers = ['o',   'P', '*', 'X', 'H', 'd']
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, curve_item in enumerate(curves_data):
        fpr_vals = np.array(curve_item['fpr'])
        tpr_vals = np.array(curve_item['tpr'])
        name = curve_item['name']
        marker_style = markers[i % len(markers)]
        current_series_color = color_cycle[i % len(color_cycle)]
        
        _plot_roc_curve_core(fpr_vals, tpr_vals, title_suffix="", save_filename=None, 
                             ax=ax, curve_label=name, plot_random_guess=False,
                             marker=marker_style, point_size=600, 
                             series_color=current_series_color) 

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('FPR', fontsize=29,labelpad=-5)
    ax.set_ylabel('TPR', fontsize=29,labelpad=-5)
    ax.tick_params(axis='both', which='major', labelsize=23)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.legend(loc=(0.42,0.017), fontsize=22)
    ax.grid(True, linestyle=':', alpha=0.9)
    ax.set_aspect('equal', adjustable='box')
    fig.savefig("experiments/effectiveness/fpr.png")
    


if __name__ == '__main__':
    
    # This is merely a plotting code. You need get the data point from CGSR.py or calculate_fprtpr.py
    curves_to_plot = [
        {
            'fpr': np.array([45/50, 41/50, 40/50, 42/50, 39/50]),
            'tpr': np.array([44/50, 42/50,42/50, 45/50, 38/50]),
            'name': 'Naive'
        },
        {
            'fpr': np.array([1.2/70, 2/70, 0.0, 3/70, 1.1/70]),
            'tpr': np.array([24/70, 19/70, 20/70, 21/70, 23/70]),
            'name': 'Opt.'
        },
        {
            'fpr': np.array([4/520, 5/520, 4/520, 6/520,3/520]),
            'tpr': np.array([429/520, 448/520, 454/520, 449/520,457/520]),
            'name': 'AQUA_acronym'
        },
        {
            'fpr': np.array([36/500, 23/500, 14/500, 25/500,30/500]),
            'tpr': np.array([406/500, 416/500, 419/500,420/500,417/500]),
            'name': 'AQUA_spatial'
        }
        
    ]
    plot_multiple_roc_curves_from_fpr_tpr(curves_to_plot)
    print("-" * 30)

    