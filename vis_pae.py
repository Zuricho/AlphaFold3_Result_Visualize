#!/usr/bin/env python3

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams["font.size"] = 12
mpl.rcParams["font.family"] = "Arial"



# helper functions

def hide_axes_frame(ax):
    """
    Hide the frame of the given matplotlib axis.
    
    Parameters:
    ax (matplotlib.axes.Axes): The axis object whose frame will be made invisible.
    """
    # Hide the axes frame
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Optionally, you might also want to hide the ticks and labels if needed
    ax.set_xticks([])
    ax.set_yticks([])


def process_token_chains(token_chain_ids):
    """
    Process token chains to assign numerical indices and find start and end indices for each unique token.
    
    Parameters:
    token_chain_ids (list): List of characters representing token chains.
    
    Returns:
    dict, dict, dict, np.ndarray: Three dictionaries mapping characters to their numerical index, start index,
                                  and end index, plus a numpy array of numerical indices.
    """
    # Convert list to numpy array if not already
    token_chain_ids = np.array(token_chain_ids)
    
    # Find unique chains and create a mapping to numerical indices
    unique_chains = np.unique(token_chain_ids)
    chain_to_num = {char: i for i, char in enumerate(unique_chains)}
    
    # Convert token chains to numerical indices
    token_chain_nums = np.array([chain_to_num[char] for char in token_chain_ids]).reshape(1, -1)
    
    # Find the start and end index for each character
    chain_to_start_index = {}
    chain_to_end_index = {}
    
    for char in unique_chains:
        indices = np.where(token_chain_ids == char)[0]
        chain_to_start_index[char] = indices[0]
        chain_to_end_index[char] = indices[-1]

    return chain_to_num, chain_to_start_index, chain_to_end_index, token_chain_nums


def main(alphafold_prediction_name):
    # get the file names
    json_name_full = [os.path.join(alphafold_prediction_name, f"{alphafold_prediction_name}_full_data_{i}.json") for i in range(5)]
    json_name_confidence = [os.path.join(alphafold_prediction_name, f"{alphafold_prediction_name}_summary_confidences_{i}.json") for i in range(5)]


    # load the data
    json_data_full = [json.load(open(i)) for i in json_name_full]
    json_data_confidence = [json.load(open(i)) for i in json_name_confidence]

    for model_num in range(5):


        json_data_full[model_num].keys()
        # atom_chain_ids, atom_plddts, contact_probs, pae, token_chain_ids, token_res_ids

        json_data_confidence[model_num].keys()
        # chain_iptm, chain_pair_iptm, chain_pair_pae_min, chain_ptm, fraction_disordered, has_clash, iptm, num_recycles, ptm, ranking_score


        # processing the data
        # chain num
        chain_to_num, chain_to_start_index, chain_to_end_index, token_chain_nums = process_token_chains(json_data_full[model_num]["token_chain_ids"])

        # get xticks data
        token_res_ids = json_data_full[model_num]["token_res_ids"]
        xticks_loc = []
        xticks_present = []
        for i in range(len(token_res_ids)):
            if token_res_ids == 1 or token_res_ids[i]%200 == 0:
                xticks_loc.append(i)
                xticks_present.append(token_res_ids[i])


        # Assuming 'json_data_full' and 'model_num' are already defined and available
        fig, ax = plt.subplots(figsize=(4, 4))

        # Display the data
        pae_array = np.array(json_data_full[model_num]["pae"])
        image = ax.imshow(pae_array, cmap="Greens_r", vmin=0, vmax=30)

        # ax.set_xticks(xticks_loc, xticks_present)
        # ax.set_xticks([])
        # ax.set_xticks(np.arange(0, len(token_res_ids), 200))
        ax.set_yticks([])
        # set the frame to dashed line
        for spine in ax.spines.values():
            spine.set_linestyle("--")
            spine.set_linewidth(1)
            spine.set_color("k")


        # Create an axes on the right side of ax, which will match the height of ax
        divider = make_axes_locatable(ax)
        ax_colorbar = divider.append_axes("right", size="5%", pad=0.2)
        ax_topbar = divider.append_axes("top", size="8%", pad=0.03)
        ax_leftbar = divider.append_axes("left", size="8%", pad=0.03)


        # topbar
        ax_topbar.imshow(token_chain_nums, cmap="tab10", aspect="auto", alpha=0.7)
        hide_axes_frame(ax_topbar)


        # leftbar
        ax_leftbar.imshow(token_chain_nums.T, cmap="tab10", aspect="auto", alpha=0.7)
        hide_axes_frame(ax_leftbar)

        # colorbar
        colorbar = fig.colorbar(image, cax=ax_colorbar, label="PAE (Å)")


        # plot a axhline at the start and end of each token
        for char, start_index in chain_to_start_index.items():
            if start_index != 0:
                ax.axhline(start_index - 0.5, color="k", linewidth=1, linestyle="--")
                ax.axvline(start_index - 0.5, color="k", linewidth=1, linestyle="--")
                ax_topbar.axvline(start_index - 0.5, color="w", linewidth=1, linestyle="-")
                ax_leftbar.axhline(start_index - 0.5, color="w", linewidth=1, linestyle="-")

        # Adding text annotations at the center of each token chain
        for char in chain_to_start_index:
            start_index = chain_to_start_index[char]
            end_index = chain_to_end_index[char]
            center_index = (start_index + end_index) / 2
            # Add text to top bar
            ax_topbar.text(center_index, 0, char, color='#222222', ha='center', va='center')
            # Add text to left bar
            ax_leftbar.text(0, center_index, char, color='#222222', ha='center', va='center')

        # Show the plot
        plt.savefig(f"{alphafold_prediction_name}/figure_pae_{model_num}.png", dpi=300, bbox_inches="tight", transparent=False)





if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vis_pae.py <alphafold_prediction_directory>")
        sys.exit(1)
    alphafold_prediction_name = sys.argv[1]
    main(alphafold_prediction_name)
