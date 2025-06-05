import matplotlib.pyplot as plt
import matplotlib.ticker 
import numpy as np

def plot_retrieval_ratios_vs_watermark_num(lines_data, 
                                           title="Retrieval Ratio", 
                                           xlabel="Number of Injected Watermarks", 
                                           ylabel="Retrieval Ratio", 
                                           save_filename="retrieval_ratios_plot.png",
                                           x_log_scale=False, 
                                           custom_xticks=None,
                                           force_y_limit=True):
    
    fig, ax = plt.subplots(figsize=(9, 9))

    markers = ['o',  'D','s','P',  '*', 'X',  'v']
    linestyles = ['-', '--', '-.', ':']
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if x_log_scale:
        ax.set_xscale('log')

    valid_lines_data_for_y = [item for item in lines_data if isinstance(item.get('retrieval_ratio'), (list, np.ndarray)) and len(item['retrieval_ratio']) > 0]

    valid_lines_data_for_x = [item for item in lines_data if isinstance(item.get('watermark_num'), (list, np.ndarray)) and len(item['watermark_num']) > 0]

    for i, line_item in enumerate(lines_data):
        current_x_vals = line_item.get('watermark_num')
        current_y_vals = line_item.get('retrieval_ratio')

        if not (isinstance(current_x_vals, (list, np.ndarray)) and len(current_x_vals) > 0 and
                isinstance(current_y_vals, (list, np.ndarray)) and len(current_y_vals) > 0):
            print(f"Warning: Skipping data series named '{line_item.get('name', 'Unknown')}' because watermark_num or retrieval_ratio is empty or incorrectly formatted.")
            continue
        
       
        if x_log_scale and not all(x > 0 for x in current_x_vals):
            print(f"Warning: Skipping data series named '{line_item.get('name', 'Unknown')}' because watermark_num contains non-positive values when x_log_scale=True.")
            continue

        x_vals = np.array(current_x_vals)
        y_vals = np.array(current_y_vals)
        name = line_item.get('name', f'Series {i+1}') 
        
        marker_style = markers[i % len(markers)]
        linestyle_style = linestyles[(i // len(markers)) % len(linestyles)] 
        current_series_color = color_cycle[i % len(color_cycle)]
        if i!=1:
            ax.plot(x_vals, y_vals, label=name, marker=marker_style, 
                    linestyle=linestyle_style, color=current_series_color, 
                    linewidth=5.5, markersize=11,alpha=0.5)
        if i==1:
            ax.plot(x_vals, y_vals, label=name, marker=marker_style, 
                    linestyle=linestyle_style, color=current_series_color, 
                    linewidth=9, markersize=15,alpha=1)
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=26)
    
    if force_y_limit:
        ax.set_ylim([0, 0.1])
    else: 
        min_y = 0
        if valid_lines_data_for_y:
            valid_ratios = [item['retrieval_ratio'] for item in valid_lines_data_for_y if isinstance(item.get('retrieval_ratio'), (list, np.ndarray)) and len(item['retrieval_ratio']) > 0]
            if valid_ratios: 
                 max_y_candidate = np.max([np.max(ratio) for ratio in valid_ratios]) * 1.1
                 max_y = max(1.0, max_y_candidate) 
            else:
                max_y = 1.0 
        else:
            max_y = 1.0 
        ax.set_ylim([min_y, max_y if max_y > min_y else 1.0])


    if not x_log_scale and valid_lines_data_for_x:
        all_x_vals_flat = [x_val for item in valid_lines_data_for_x for x_val in item['watermark_num']]
        if all_x_vals_flat: 
            min_x_data = np.min(all_x_vals_flat)
            max_x_data = np.max(all_x_vals_flat)
            x_padding = (max_x_data - min_x_data) * 0.05 if max_x_data > min_x_data else 1.0 
            ax.set_xlim([min_x_data - x_padding, max_x_data + x_padding])

    if custom_xticks:
        ax.set_xticks(custom_xticks)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()) 
        if x_log_scale:
            ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())


    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        
    if any(isinstance(line_item.get('watermark_num'), (list, np.ndarray)) and len(line_item['watermark_num']) > 0 and \
           isinstance(line_item.get('retrieval_ratio'), (list, np.ndarray)) and len(line_item['retrieval_ratio']) > 0 and \
           (not x_log_scale or all(x > 0 for x in line_item['watermark_num']))
           for line_item in lines_data):
        ax.legend(loc="best", fontsize=30) 
        
    ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout() 
    fig.savefig(save_filename, dpi=300)
    

if __name__ == '__main__':
   # You need to obtain the data from calculate_retrieval_ratio.py
    lines_to_plot_log_x = [
        {
            'watermark_num': np.array([1, 10, 100,  1000,  10000]), 
            'retrieval_ratio': np.array([0.0003, 0.0005, 0.0012, 0.021, 0.092]),
            
            'name': 'Naive' 
        },
        {
            'watermark_num': np.array([1, 10, 100, 1000, 10000]), 
            'retrieval_ratio': np.array([0.0001102, 0.0001102, 0.0001102, 0.0001102, 0.0001102]),
            'name': 'AQUA_acronym' 
        },
        { 
            'watermark_num': np.array([1, 10, 100, 1000, 10000]),
            'retrieval_ratio': np.array([0.0001102, 0.0001102, 0.0003102, 0.0007102, 0.0009]), 
            'name': 'AQUA_spatial'
        }
        
    ]
    
    user_custom_xticks = [1, 10, 100, 1000, 10000] 
    
    plot_retrieval_ratios_vs_watermark_num(
        lines_to_plot_log_x,
        title="Stealthiness of Normal Queries",
        xlabel="Number of Injected Watermarks",
        ylabel="Retrieval Rate", 
        save_filename="/home/cty/WatermarkmmRAG/experiments/stealthiness/watermark_num.png",
        x_log_scale=True, 
        custom_xticks=user_custom_xticks,
        force_y_limit=True
    )
    print("-" * 30)
