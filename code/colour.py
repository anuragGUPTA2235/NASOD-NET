import matplotlib.colors as mcolors

# Given color list
label_clrs = [
    "#ff0000", "#2e8b57", "#808000", "#800000", "#000080", "#2f4f4f", "#ffa500",
    "#00ff00", "#ba55d3", "#00fa9a", "#00ffff", "#0000ff", "#f08080", "#ff00ff",
    "#1e90ff", "#ffff54", "#dda0dd", "#ff1493", "#87cefa", "#ffe4c4"
]

# Number of colors needed
num_colors_needed = 91

# Calculate the number of times to repeat the original list
repeat_times = num_colors_needed // len(label_clrs) + 1

# Repeat the original list
extended_clrs = label_clrs * repeat_times

# Get evenly spaced colors from the extended list
extended_clrs = extended_clrs[:num_colors_needed]

# Check if there are any duplicate colors
if len(set(extended_clrs)) != len(extended_clrs):
    # If duplicates exist, generate additional colors
    additional_colors_needed = num_colors_needed - len(extended_clrs)
    additional_colors = list(mcolors.CSS4_COLORS.keys())[:additional_colors_needed]
    extended_clrs.extend(additional_colors)

# Display the extended list of colors
print(extended_clrs)
