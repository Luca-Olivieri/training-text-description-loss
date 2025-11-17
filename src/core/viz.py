"""
Visualization and analysis utilities for model evaluation and data processing.

This module provides comprehensive tools for visualizing and analyzing machine learning
models, particularly focused on semantic segmentation and vision-language tasks. It includes:

- Display and rendering utilities for prompts, images, and metrics
- Attention and similarity map visualization with overlay capabilities
- Performance metrics computation and formatting
- HTML gallery generation for multi-row image comparisons
- Data analysis tools for computing statistics from experiment results
- Class distribution analysis for segmentation datasets
- Mask creation and manipulation utilities for class-specific visualizations

The module integrates with PyTorch, xarray, pandas, and various visualization libraries
to provide end-to-end support for model evaluation workflows, especially in Jupyter notebooks.
"""

from core.config import *
from core.prompter import Prompt, get_significant_classes
from core.color_map import apply_colormap
from core.torch_utils import blend_tensors
from core.datasets import JsonlIO, get_answer_objects, VOC2012SegDataset, SegDataset
from core.data_utils import flatten_list_of_lists
from core.pipeline import format_pos_ovr_masks
from models.vle import VLEncoder, MapComputeMode, VLEImageOutput, VLETextOutput

from IPython.display import Markdown, display

import PIL
from PIL import Image
import seaborn as sns
import xarray as xr
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
import base64
from tqdm import tqdm

import torch
from torch import nn
import torchmetrics as tm
from torch.utils.data import DataLoader
from torchvision.transforms.v2.functional import to_pil_image
import torchvision.transforms.v2.functional as TF
import torch.nn.functional as F

from core._types import Iterable, Literal, Optional

def print_file_content(
        filename: str
) -> None:
    """
    Prints the entire content of a file to stdout.

    Reads and displays the file line by line, preserving original formatting.
    Prints a warning message if the file does not exist.

    Args:
        filename (str): Path to the file to print.

    Returns:
        None: Prints directly to stdout.

    Example:
        >>> print_file_content("config.yml")
        --- Contents of 'config.yml' ---
        ...file contents...
        --- End of file ---
    """
    if not os.path.exists(filename):
        print(f"File '{filename}' does not exist.")
        return
        
    print(f"--- Contents of '{filename}' ---")
    with open(filename, 'r') as f:
        for line in f:
            print(line, end='')
    print("--- End of file ---\n")

def my_tqdm(
        data: Iterable,
        desc: str = ""
) -> tqdm:
    """
    Wraps an iterable with a customized tqdm progress bar.

    Creates a styled progress bar with enumeration support. Converts non-list
    iterables to lists to enable proper progress tracking. The progress bar
    includes custom formatting with a green color scheme and detailed time estimates.

    Args:
        data (Iterable): The data to wrap with a progress bar. Will be converted
            to a list if not already a list.
        desc (str, optional): Description text displayed before the progress bar.
            Defaults to an empty string.

    Returns:
        tqdm: An enumerated tqdm progress bar iterator that yields (index, item) tuples.

    Example:
        >>> for idx, item in my_tqdm(data_list, desc="Processing"):
        ...     process(item)
    """
    if not isinstance(data, list):
        data = list(data)
    return tqdm(
        enumerate(data),
        total=len(data),
        desc=desc, # Add a description
        unit="item", # Specify the unit of progress
        colour="#67ad5b", # Set a vibrant color,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

def pretty_metrics(
        metric_collection: tm.MetricCollection
) -> dict:
    """
    Formats a torchmetrics MetricCollection into human-readable strings.

    Converts metric values to formatted strings with appropriate precision.
    Learning rate metrics are formatted in scientific notation, while other
    metrics use fixed-point notation with 4 decimal places.

    Args:
        metric_collection (tm.MetricCollection): A collection of computed metrics
            from torchmetrics.

    Returns:
        dict: A dictionary mapping metric names to formatted string values.

    Example:
        >>> metrics = MetricCollection([Accuracy(), F1Score()])
        >>> pretty_metrics(metrics)
        {'accuracy': '0.9234', 'f1_score': '0.8765'}
    """
    return {m: f"{s.item():.4f}" if 'lr' not in m else f"{s.item():.2e}" for m, s in metric_collection.items()}

def normalize_attn_maps(
        attn_maps: torch.Tensor,        
) -> torch.Tensor:
    """
    Normalizes attention maps to the range [0, 1] using min-max normalization.

    Performs per-image normalization by computing min and max values over the
    last two dimensions (typically height and width). This ensures each attention
    map in a batch is independently normalized.

    Args:
        attn_maps (torch.Tensor): Attention maps tensor of any shape. Normalization
            is applied over the last two dimensions.

    Returns:
        torch.Tensor: Normalized attention maps with values in [0, 1], same shape
            as input.

    Example:
        >>> attn = torch.rand(4, 8, 16, 16)  # batch, heads, H, W
        >>> norm_attn = normalize_attn_maps(attn)
        >>> assert norm_attn.min() >= 0 and norm_attn.max() <= 1
    """
    norm_dims = list(range(attn_maps.ndim))[-2:] # indices of the last two dimensions (per-image normalization)
    max_per_image = attn_maps.amax(dim=norm_dims, keepdim=True)
    min_per_image = attn_maps.amin(dim=norm_dims, keepdim=True)
    attn_maps = (attn_maps - min_per_image)/(max_per_image - min_per_image)
    return attn_maps

def normalize_sim_maps(
        sim_maps: torch.Tensor,        
) -> torch.Tensor:
    """
    Normalizes similarity maps preserving sign information.

    Handles both positive and negative similarity values by separately normalizing
    each using the absolute maximum value per map. This preserves the sign while
    scaling values to the range [-1, 1].

    Args:
        sim_maps (torch.Tensor): Similarity maps that may contain both positive
            and negative values. Normalization is applied over the last two dimensions.

    Returns:
        torch.Tensor: Normalized similarity maps with values in [-1, 1], same shape
            as input. Sign information is preserved.

    Note:
        Positive and negative values are normalized independently to ensure
        the full dynamic range is utilized while maintaining interpretability.

    Example:
        >>> sim = torch.randn(4, 16, 16)  # Can have positive and negative values
        >>> norm_sim = normalize_sim_maps(sim)
        >>> assert norm_sim.abs().max() <= 1
    """
    pos_sim_maps = sim_maps.clamp(min=0)
    neg_sim_maps = -(sim_maps.clamp(max=0))
    norm_dims = list(range(sim_maps.ndim))[-2:] # indices of the last two dimensions (per-image normalization)

    abs_max_per_map = sim_maps.abs().amax(dim=norm_dims, keepdim=True)

    if torch.any(pos_sim_maps != 0):
        pos_sim_maps = pos_sim_maps/abs_max_per_map
    
    if torch.any(neg_sim_maps != 0):
        neg_sim_maps = neg_sim_maps/abs_max_per_map
        neg_sim_maps = -neg_sim_maps

    sim_maps = pos_sim_maps + neg_sim_maps

    return sim_maps

def overlay_map(
        background: torch.Tensor | Image.Image,
        map: torch.Tensor,
        alpha: float = 1.0,
        pos_rgb_fill: tuple[int, int, int] = (0, 255, 0),
        neg_rgb_fill: tuple[int, int, int] = (0, 0, 255),
        normalize: bool = True
) -> Image.Image:
    """
    Creates an overlay visualization of a signed map on a background image.

    Overlays positive and negative regions of a map (e.g., similarity, attention)
    onto a background image with different colors. Positive values are shown with
    pos_rgb_fill color (default green) and negative values with neg_rgb_fill color
    (default blue). The overlay opacity is controlled by the absolute map values
    and the alpha parameter.

    Args:
        background (torch.Tensor | Image.Image): The background image. If a tensor,
            should have shape (C, H, W). Will be converted to RGBA PIL Image.
        map (torch.Tensor): The signed map to overlay, shape (H, W). Positive and
            negative values are colored differently.
        alpha (float, optional): Overall opacity multiplier for the overlay.
            Defaults to 1.0.
        pos_rgb_fill (tuple[int, int, int], optional): RGB color (0-255) for positive
            values. Defaults to (0, 255, 0) (green).
        neg_rgb_fill (tuple[int, int, int], optional): RGB color (0-255) for negative
            values. Defaults to (0, 0, 255) (blue).
        normalize (bool, optional): Whether to normalize the map using normalize_sim_maps.
            Defaults to True.

    Returns:
        Image.Image: An RGB PIL Image with the map overlaid on the background.

    Example:
        >>> img = torch.rand(3, 224, 224)
        >>> sim_map = torch.randn(224, 224)
        >>> result = overlay_map(img, sim_map, alpha=0.5)
    """
    if isinstance(background, torch.Tensor):
        background_img = to_pil_image(background.cpu()).convert('RGBA')
    else:
        background_img = background.convert('RGBA')
    
    pos_mask = (map > 0).cpu() # Create a mask where overlay > 0 is True (positive), else False (negative)
    mask_expanded = pos_mask.unsqueeze(0) # Unsqueeze the mask to [1, H, W] to allow broadcasting with [3, 1, 1] for RGB values

    # Unsqueeze the RGB fill values to [3, 1, 1]
    pos_rgb_fill_expanded = (torch.tensor(pos_rgb_fill)/255.).unsqueeze(1).unsqueeze(2)
    neg_rgb_fill_expanded = (torch.tensor(neg_rgb_fill)/255.).unsqueeze(1).unsqueeze(2)

    # Use torch.where to select values based on the mask
    # This will broadcast mask_expanded to [3, H, W] and then apply the condition
    output_rgb = torch.where(mask_expanded, pos_rgb_fill_expanded, neg_rgb_fill_expanded).squeeze(0)
    
    if normalize:
        map = normalize_sim_maps(map)

    overlay_tensor = torch.concat([output_rgb.cpu(), map.abs().cpu()*(alpha)], dim=0) # [4, H, W]
    overlay_img = to_pil_image(overlay_tensor).convert('RGBA')
    return Image.alpha_composite(background_img, overlay_img).convert('RGB')

def display_token_length_distr(
        token_lengths: list[int],
        bins: int = 20
) -> None:
    """
    Displays a histogram of token length distribution.

    Creates a histogram visualization using seaborn to show the distribution
    of token lengths in a dataset. Useful for analyzing text data characteristics.

    Args:
        token_lengths (list[int]): A list of token length values to plot.
        bins (int, optional): Number of bins for the histogram. Defaults to 20.

    Returns:
        None: Displays the plot using matplotlib.

    Example:
        >>> lengths = [10, 15, 12, 18, 20, 15, 13]
        >>> display_token_length_distr(lengths, bins=10)
    """
    sns.histplot(token_lengths, bins=bins)
    plt.xlabel("Token Length")
    plt.ylabel("Count")
    plt.title("Distribution of Token Lengths")
    plt.show()

def display_prompt(full_prompt: str | Prompt) -> None:
    """
    Displays a prompt in a Jupyter notebook using IPython display utilities.

    Handles both simple string prompts and complex multi-modal prompts that
    contain mixtures of text and images. Text is rendered as Markdown, and
    images are displayed inline.

    Args:
        full_prompt (str | Prompt): The prompt to display. Can be either:
            - A simple string (displayed as Markdown)
            - A Prompt object (list) containing strings and/or PIL Images

    Returns:
        None: Renders the prompt directly in the notebook.

    Example:
        >>> display_prompt("# This is a title\\nSome text")
        >>> display_prompt([img1, "Description", img2])
    """
    if isinstance(full_prompt, str):
        display(Markdown(full_prompt))
    else:
        for prompt in full_prompt:
            if isinstance(prompt, Image.Image):
                display(prompt)
            else:
                display(Markdown(prompt))
    
def write_html_multi_row_image_caption(
        title: str,
        rows: dict[str, list[Image.Image]],
        captions: list[str]
) -> None:
    """
    Writes an HTML file containing a multi-row image gallery with captions.

    Creates an interactive HTML gallery where images are organized in rows
    (one row per key in the rows dict) and columns (one column per caption).
    The gallery scrolls horizontally and displays all images with their
    corresponding captions. The container sizes scale with the image sizes.

    Args:
        title (str): The title for the HTML page and the output filename.
        rows (dict[str, list[Image.Image]]): A dictionary where keys are row
            labels and values are lists of PIL Images for that row. All lists
            must have the same length.
        captions (list[str]): A list of caption strings, one per column.
            Length must match the length of image lists in rows.

    Returns:
        None: Writes an HTML file named "{title}.html" to the current directory.

    Raises:
        IOError: If the file cannot be written.

    Example:
        >>> from PIL import Image
        >>> img1 = Image.new('RGB', (100, 50), color = 'red')
        >>> img2 = Image.new('RGB', (120, 50), color = 'green')
        >>> img3 = Image.new('RGB', (100, 80), color = 'blue')
        >>> img4 = Image.new('RGB', (120, 90), color = 'yellow')
        >>> rows = {"Row A": [img1, img2], "Row B": [img3, img4]}
        >>> captions = ["Caption 1", "Caption 2"]
        >>> write_html_multi_row_image_caption("Dynamic_Gallery", rows, captions)
    """
    row_labels = list(rows.keys())
    # Transpose the data from row-major to column-major for the gallery function
    column_data = [list(x) for x in list(zip(*list(rows.values())))]

    generated_html = create_multi_row_gallery(title, row_labels, column_data, captions)

    output_filename = f"{title}.html"
    try:
        with open(output_filename, "w", encoding="utf-8") as file:
            file.write(generated_html)
        print(f"Successfully created {output_filename}. Open it in your browser to see the result.")
    except IOError as e:
        print(f"Error writing to file: {e}")

def create_multi_row_gallery(
        title: str,
        row_labels: list[str],
        column_data: list[list[Image.Image]],
        captions: list[str]
) -> str:
    """
    Generates HTML for a multi-row image gallery with dynamically sized cells.

    Creates a styled HTML page with a grid layout where the size of each column
    and row is determined by the largest image within it. The gallery uses
    Tailwind CSS for styling and supports horizontal scrolling.

    Args:
        title (str): The page title displayed at the top and in the browser tab.
        row_labels (list[str]): A list of strings for the row headers.
        column_data (list[list[Image.Image]]): A list where each inner list contains
            PIL Image objects for one vertical column.
        captions (list[str]): A list of captions, one for each column.

    Returns:
        str: The complete HTML content as a string.
    """
    # --- Data Validation ---
    num_columns = len(column_data)
    num_rows = len(row_labels)
    if num_columns != len(captions):
        print("Error: Number of columns must match number of captions.")
        return "Error page: data mismatch."
    for i, col in enumerate(column_data):
        if len(col) != num_rows:
            print(f"Error: Column {i+1} has {len(col)} images, but there are {num_rows} row labels.")
            return "Error page: data mismatch."

    # --- Pre-calculate dimensions and encode images ---
    image_details = [] # Stores (base64_src, width, height) for each image
    max_column_widths = [0] * num_columns
    max_row_heights = [0] * num_rows

    for col_idx, col in enumerate(column_data):
        col_details = []
        for row_idx, image in enumerate(col):
            base64_src, _ = get_image_base64_data(image)
            width, height = image.size if image else (0, 0)
            
            col_details.append({'src': base64_src, 'width': width, 'height': height})

            # Update max dimensions
            if width > max_column_widths[col_idx]:
                max_column_widths[col_idx] = width
            if height > max_row_heights[row_idx]:
                max_row_heights[row_idx] = height
        image_details.append(col_details)

    # --- Generate CSS for the dynamic grid ---
    # Add a small buffer (16px) for padding (p-2 is 8px on each side)
    grid_cols_css = ' '.join([f'{w + 16}px' for w in max_column_widths])
    grid_rows_css = ' '.join([f'{h + 16}px' for h in max_row_heights])
    
    # --- Grid Generation ---
    grid_items_html = []
    
    # Generate image rows
    for row_idx, label_text in enumerate(row_labels):
        grid_items_html.append(f"""
        <div class="sticky left-0 bg-gray-100 p-4 flex items-center justify-end z-10">
            <span class="font-bold text-gray-600 text-right">{label_text}</span>
        </div>""")

        for col_idx in range(num_columns):
            details = image_details[col_idx][row_idx]
            if details['src']:
                grid_items_html.append(f"""
                <div class="bg-white shadow-lg rounded-lg overflow-hidden flex items-center justify-center p-2">
                    <img src="{details['src']}" alt="Image for {captions[col_idx]}" class="max-w-full max-h-full object-contain">
                </div>""")
            else:
                grid_items_html.append('<div class="bg-gray-200 shadow-lg rounded-lg flex items-center justify-center"><span class="text-xs text-gray-500">Image not found</span></div>')

    # Generate caption row
    grid_items_html.append('<div class="sticky left-0 bg-gray-100 z-10"></div>')
    for caption_text in captions:
        grid_items_html.append(f"""
        <figcaption class="pt-2 text-gray-700 text-sm leading-relaxed">
            {caption_text}
        </figcaption>""")

    # --- HTML and CSS Generation ---
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #f3f4f6; /* bg-gray-100 */
        }}
        .horizontal-scroll-container {{
            scrollbar-width: thin;
            scrollbar-color: #9ca3af #e5e7eb;
        }}
        .horizontal-scroll-container::-webkit-scrollbar {{ height: 12px; }}
        .horizontal-scroll-container::-webkit-scrollbar-track {{ background: #e5e7eb; }}
        .horizontal-scroll-container::-webkit-scrollbar-thumb {{
            background-color: #9ca3af;
            border-radius: 10px;
            border: 3px solid #f3f4f6;
        }}
        .horizontal-scroll-container::-webkit-scrollbar-thumb:hover {{ background: #6b7280; }}
        
        .gallery-grid {{
            display: inline-grid;
            grid-auto-flow: row;
            /* Dynamically set column and row sizes */
            grid-template-columns: 10rem {grid_cols_css};
            grid-template-rows: {grid_rows_css} auto;
            gap: 1rem;
        }}
    </style>
</head>
<body class="min-h-screen flex flex-col items-center p-4 sm:p-8">

    <div class="w-full max-w-screen-xl mx-auto">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-8">{title}</h1>
        
        <div class="horizontal-scroll-container overflow-x-auto pb-4">
            <div class="gallery-grid">
                {''.join(grid_items_html)}
            </div>
        </div>
    </div>

</body>
</html>
"""
    return html_content

def get_image_base64_data(image):
    """
    Converts a PIL Image object to a Base64 encoded data URI string.

    Encodes an image as PNG format and converts it to a base64 string suitable
    for embedding in HTML. Handles various image modes by converting to RGB.

    Args:
        image (Image.Image | None): A PIL Image object to encode. If None,
            returns (None, None).

    Returns:
        tuple[str | None, str | None]: A tuple containing:
            - The base64 data URI string (e.g., "data:image/png;base64,...")
            - The MIME type string ("image/png")
            Returns (None, None) if the image is None or encoding fails.

    Note:
        Prints warning messages to stdout if the image is None or encoding fails.

    Example:
        >>> img = Image.open("photo.jpg")
        >>> data_uri, mime = get_image_base64_data(img)
        >>> # Use in HTML: <img src="{data_uri}">
    """
    if image is None:
        print("Warning: Image is None. Skipping.")
        return None, None

    try:
        from io import BytesIO
        
        # Convert image to RGB if it's not already (handles RGBA, L, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save image to BytesIO buffer as PNG
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Encode to base64
        encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}", "image/png"
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None, None

def compute_results_da(
        exp_path: Path,
        jsonlio: JsonlIO
) -> xr.DataArray:
    """
    Computes evaluation results as an xarray DataArray from experiment JSONL files.

    Reads all JSONL files in the specified experiment directory, extracts evaluation
    predictions, scores, and reasons, and organizes them into a multi-dimensional
    xarray DataArray for easy analysis and visualization.

    Args:
        exp_path (Path): Path to the experiment directory containing JSONL result files.
        jsonlio (JsonlIO): A JsonlIO instance for reading and parsing JSONL files.

    Returns:
        xr.DataArray: A 3D DataArray with dimensions ["var", "img_idx", "metric"].
            - "var": Variable names derived from JSONL filenames
            - "img_idx": Image indices
            - "metric": ["pred", "reason", "score"] evaluation metrics

    Example:
        >>> results = compute_results_da(Path("experiments/exp1"), jsonlio)
        >>> accuracy = (results.sel(metric="pred") == 1).mean()
    """
    data_da = None

    var_paths = glob(f"{exp_path}/*.jsonl")
    var_names = [os.path.splitext(os.path.basename(path))[0] for path in var_paths]

    for var_n, var_p in zip(var_names, var_paths):
        
        eval_prs = get_answer_objects(var_p, idxs=None, jsonlio=jsonlio, return_state=False, format_to_dict=True)
        # eval_prs = get_many_eval_pr(var_p, return_state=False, format_to_dict=True)
        prs_per_img_idx_df = pd.DataFrame.from_dict(eval_prs, orient='index')
        prs_per_img_idx_df = prs_per_img_idx_df.sort_index().sort_index(axis=1)

        prs_per_img_idx_pred_df = (prs_per_img_idx_df["pred"] == "correct").astype(float)
        prs_per_img_idx_score_df = prs_per_img_idx_df["score"]
        prs_per_img_idx_reason_df = prs_per_img_idx_df["reason"]

        prs_per_img_idx_da = xr.DataArray(
            [prs_per_img_idx_pred_df, prs_per_img_idx_reason_df, prs_per_img_idx_score_df],
            coords=[["pred", "reason", "score"], prs_per_img_idx_df.index],
            dims=["metric", "img_idx"]
        ).transpose("img_idx", "metric")
        
        if data_da is None:
            coords = [var_names, prs_per_img_idx_df.index, ["pred", "reason", "score"]] # indexes names
            sorted_coords = [sorted(dim_values) for dim_values in coords]
            dims = ["var", "img_idx", "metric"] # dimensions names
            shape = [len(l) for l in sorted_coords]
            data_da = xr.DataArray(np.empty(shape, dtype=object), coords=sorted_coords, dims=dims)

        data_da.loc[var_n] = prs_per_img_idx_da

    return data_da

def compute_results_da_class_splitted(
        exp_path: Path,
        num_classes: int,
        jsonlio: JsonlIO
) -> xr.DataArray:
    """
    Computes class-specific evaluation results as an xarray DataArray.

    Similar to compute_results_da but handles results that are split by class.
    Reads JSONL files containing per-class evaluation results and organizes them
    into a 4D DataArray with an additional dimension for class labels.

    Args:
        exp_path (Path): Path to the experiment directory containing JSONL result files.
        num_classes (int): Total number of classes in the dataset. Used to ensure
            all class columns are present even if some are missing data.
        jsonlio (JsonlIO): A JsonlIO instance for reading and parsing JSONL files.

    Returns:
        xr.DataArray: A 4D DataArray with dimensions ["var", "img_idx", "pos_class", "metric"].
            - "var": Variable names derived from JSONL filenames
            - "img_idx": Image indices
            - "pos_class": Class labels (0 to num_classes-1)
            - "metric": ["pred", "reason", "score"] evaluation metrics

    Example:
        >>> results = compute_results_da_class_splitted(Path("exp"), 21, jsonlio)
        >>> class_5_acc = (results.sel(pos_class=5, metric="pred") == 1).mean()
    """
    data_da = None

    var_paths = glob(f"{exp_path}/*.jsonl")
    var_names = [os.path.splitext(os.path.basename(path))[0] for path in var_paths]

    for var_n, var_p in zip(var_names, var_paths):
        
        eval_prs = get_answer_objects(var_p, idxs=None, jsonlio=jsonlio, return_state=False, format_to_dict=True)
        prs_per_img_idx_df = pd.DataFrame.from_dict(eval_prs, orient='index')
        
        for column in [str(n) for n in range(0, num_classes)]:
            if column not in prs_per_img_idx_df.columns:
                prs_per_img_idx_df[column] = pd.NA
        prs_per_img_idx_df.columns = [int(s) for s in prs_per_img_idx_df.columns]
        prs_per_img_idx_df = prs_per_img_idx_df.sort_index().sort_index(axis=1)

        prs_per_img_idx_pred_df = prs_per_img_idx_df.map(lambda x: x["pred"] == "correct" if type(x) == dict else None).astype(float)
        prs_per_img_idx_score_df = prs_per_img_idx_df.map(lambda x: x["score"] if type(x) == dict else None)
        prs_per_img_idx_reason_df = prs_per_img_idx_df.map(lambda x: x["reason"] if type(x) == dict else None)

        prs_per_img_idx_da = xr.DataArray(
            [prs_per_img_idx_pred_df, prs_per_img_idx_reason_df, prs_per_img_idx_score_df],
            coords=[["pred", "reason", "score"], prs_per_img_idx_df.index, prs_per_img_idx_df.columns],
            dims=["metric", "img_idx", "pos_class"]
        ).transpose("img_idx", "pos_class", "metric")
        
        if data_da is None:
            coords = [var_names, prs_per_img_idx_df.index, prs_per_img_idx_df.columns, ["pred", "score", "reason"]] # indexes names
            sorted_coords = [sorted(dim_values) for dim_values in coords]
            dims = ["var", "img_idx", "pos_class", "metric"] # dimensions names
            shape = [len(l) for l in sorted_coords]
            data_da = xr.DataArray(np.empty(shape, dtype=object), coords=sorted_coords, dims=dims)

        data_da.loc[var_n] = prs_per_img_idx_da

    return data_da

def describe_da(
        data_da: xr.DataArray,
        dims_to_agg: list[str]
) -> pd.DataFrame:
    """
    Computes descriptive statistics for an xarray DataArray.

    Aggregates data along specified dimensions and computes mean, standard
    deviation, minimum, and maximum values. Returns results as a pandas DataFrame.

    Args:
        data_da (xr.DataArray): The data array to compute statistics for.
        dims_to_agg (list[str]): List of dimension names to aggregate over.

    Returns:
        pd.DataFrame: A DataFrame with statistics (mean, std, min, max) as columns.
            Remaining dimensions from the DataArray become rows/indices.

    Raises:
        AttributeError: If the resulting DataFrame has more than 2 dimensions.

    Example:
        >>> da = xr.DataArray(np.random.rand(10, 5), dims=["img", "class"])
        >>> stats = describe_da(da, dims_to_agg=["img"])
        >>> # Returns stats aggregated over images, indexed by class
    """
    data_da = data_da.astype("float")
    stats = {
        "mean": data_da.mean(dim=dims_to_agg),
        "std": data_da.std(dim=dims_to_agg, ddof=1),
        "min": data_da.min(dim=dims_to_agg),
        "max": data_da.max(dim=dims_to_agg),
    }
    # Convert to DataArrays and stack
    df = xr.concat(stats.values(), dim="stat").assign_coords(stat=list(stats)).to_pandas().transpose()
    if len(df.shape) > 2: raise AttributeError 
    return df

def flatten_class_splitted_answers(
        class_splitted_answers: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Flattens class-splitted answer dictionaries into a single list.

    Converts nested answer structures where each entry contains multiple
    class-specific answers into a flat list where each answer has its own
    entry with a unique image index.

    Args:
        class_splitted_answers (list[dict[str, Any]]): A list of dictionaries,
            each containing a "content" key with a dict of class-specific answers.

    Returns:
        list[dict[str, Any]]: A flattened list where each dictionary contains
            "img_idx" (sequential index) and "content" (the answer).

    Example:
        >>> answers = [{"content": {"0": "ans1", "1": "ans2"}}]
        >>> flatten_class_splitted_answers(answers)
        [{"img_idx": 0, "content": "ans1"}, {"img_idx": 1, "content": "ans2"}]
    """
    flat_answers = []
    b = 0
    for csa in class_splitted_answers:
        new_answers = list(csa["content"].values())
        new_answers = [{"img_idx": b+i, "content": a} for i, a in enumerate(new_answers)]
        flat_answers.extend(new_answers)
        b += len(new_answers)
    return flat_answers

def class_pixel_distribution(
        dl: DataLoader,
        num_classes: int,
        device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the pixel-wise class distribution for a segmentation dataset.

    Iterates through a DataLoader and counts the number of pixels belonging to
    each class across all images. Returns both absolute counts and percentages.

    Args:
        dl (DataLoader): PyTorch DataLoader providing (image, label) batches.
            Labels are expected to be 2D tensors (H, W) with integer class IDs.
        num_classes (int): The total number of classes in the dataset.
        device (torch.device): The device to perform computations on.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - class_pixel_counts (torch.Tensor): A 1D tensor of shape (num_classes,)
                where each element is the total count of pixels for that class.
            - class_pixel_distribution_percentage (torch.Tensor): A 1D tensor of
                shape (num_classes,) with the percentage of pixels for each class.

    Raises:
        ValueError: If num_classes is not a positive integer.

    Note:
        Prints a warning if no pixels are processed (empty dataset).

    Example:
        >>> counts, percentages = class_pixel_distribution(train_loader, 21, device)
        >>> print(f"Class 0: {percentages[0]:.2f}% of pixels")
    """
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("'num_classes' must be a positive integer.")

    # Initialize a tensor to store pixel counts for each class
    class_pixel_counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    # Iterate through the DataLoader
    for i, (_, labels) in enumerate(dl):
        # labels will be of shape [batch_size, H, W]
        # For each label in the batch:
        for label_map in labels:
            # label_map is [H, W]
            # Flatten the label map to easily count pixel values
            flattened_label = label_map.view(-1)

            # Use torch.bincount to count occurrences of each class ID
            # minlength ensures the tensor has 'num_classes' elements even if some classes are missing
            counts = torch.bincount(flattened_label, minlength=num_classes)

            # Add these counts to our total
            class_pixel_counts += counts
        
    total_pixels = torch.sum(class_pixel_counts).item()

    if total_pixels == 0:
        print("Warning: No pixels processed. Check your dataset and labels.")
        return class_pixel_counts, torch.zeros(num_classes, dtype=torch.float)
    else:
        # Calculate percentages
        class_pixel_distribution_percentage = (class_pixel_counts.float() / total_pixels) * 100
        return class_pixel_counts, class_pixel_distribution_percentage

def get_layer_numel_str(
        module: nn.Module,
        print_only_total: bool = False,
        only_trainable: bool = False
) -> str:
    """
    Returns a formatted string showing parameter counts for a PyTorch module.

    Generates a human-readable summary of the number of parameters in each layer
    of a neural network module. Can optionally filter to show only trainable
    parameters or only the total count.

    Args:
        module (nn.Module): The PyTorch module to analyze.
        print_only_total (bool, optional): If True, only returns the total count.
            If False, includes per-layer breakdown. Defaults to False.
        only_trainable (bool, optional): If True, only counts parameters with
            requires_grad=True. If False, counts all parameters. Defaults to False.

    Returns:
        str: A formatted string with parameter counts. Numbers are comma-separated
            for readability. Returns empty string if no parameters found.

    Example:
        >>> model = torchvision.models.resnet18()
        >>> print(get_layer_numel_str(model, only_trainable=True))
        conv1.weight: 9,408
        ...
        Total: 11,689,512
    """
    s = 0
    out_str: str = ""
    layer_names = []
    layer_params = []
    for name, params in module.named_parameters():
        if (only_trainable and params.requires_grad) or not only_trainable:
            layer_names.append(name)
            layer_params.append(params)
            s += params.numel()
    if len(layer_names) > 0:
        max_name_len = max([len(n) for n in layer_names])
        if not print_only_total:
            out_str += '\n'.join([f"{n:<{max_name_len+1}}: {ps.numel():,}" for n, ps in zip(layer_names, layer_params)]) + '\n'
        out_str += f"Total: {s:,}"
    return out_str

def format_to_title(
        text: str,
        total_length: int = 100,
        pad_symbol: str = '-'
) -> str:
    """
    Centers a title within a given total length, padding with a specified symbol.

    Wraps the text in square brackets and centers it within the specified total
    length by padding with the pad_symbol character. If the text is too long,
    returns it wrapped in brackets without padding.

    Args:
        text (str): The string to be centered.
        total_length (int, optional): The total desired length of the output string.
            Defaults to 100.
        pad_symbol (str, optional): The character to use for padding. Defaults to '-'.

    Returns:
        str: The centered string with padding. If total_length is less than or equal
            to the length of "[ text ]", returns just "[ text ]" without padding.

    Example:
        >>> format_to_title("Results", total_length=20, pad_symbol='-')
        '-----[ Results ]-----'
    """
    text = f"[ {text} ]"
    if total_length <= len(text):
        return text

    padding_needed = total_length - len(text)
    
    # Calculate padding for left and right
    # Integer division might put an extra '-' on the right if padding_needed is odd
    left_padding = padding_needed // 2
    right_padding = padding_needed - left_padding

    centered_string = pad_symbol * left_padding + text + pad_symbol * right_padding
    return centered_string


def create_diff_mask(
        mask1: torch.Tensor,
        mask2: torch.Tensor,
) -> torch.Tensor:
    """
    Creates a binary difference mask from two integer-based segmentation masks.

    Compares two segmentation masks element-wise and produces a binary mask
    indicating where they differ. The operation is fully vectorized and runs
    efficiently on CUDA devices.

    Args:
        mask1 (torch.Tensor): The first segmentation mask with class indices.
            Expected dtype: torch.long, torch.int, or similar integer type.
        mask2 (torch.Tensor): The second segmentation mask with class indices.
            Must have the same shape and device as mask1.

    Returns:
        torch.Tensor: A binary mask with dtype uint8. Value is 1 where pixels
            in mask1 and mask2 are different, and 0 where they are the same.

    Raises:
        AssertionError: If mask1 and mask2 have different shapes.

    Example:
        >>> gt_mask = torch.randint(0, 21, (224, 224))
        >>> pred_mask = torch.randint(0, 21, (224, 224))
        >>> diff = create_diff_mask(gt_mask, pred_mask)
        >>> error_rate = diff.float().mean()
    """
    # 1. Ensure the masks have the same shape for element-wise comparison.
    assert mask1.shape == mask2.shape, f"Input masks must have the same shape, but got {mask1.shape} and {mask2.shape}"

    # 2. Perform element-wise comparison. This creates a boolean tensor.
    #    'True' where elements are not equal, 'False' where they are equal.
    #    This is the functional equivalent of `torch.ne(mask1, mask2)`.
    diff = (mask1 != mask2).to(torch.uint8)

    return diff

def create_cs_masks_(
        sc_img: torch.Tensor,
        mask: torch.Tensor,
        sign_classes: list[int],
        alpha: float = 0.55
) -> dict[int, torch.Tensor]:
    """
    Creates class-specific overlay masks in a vectorized, GPU-friendly manner.

    For each class in sign_classes, creates a binary mask highlighting pixels
    belonging to that class, then blends it with the source image to create
    a colored overlay visualization. All operations are vectorized for efficiency.

    Args:
        sc_img (torch.Tensor): Source image tensor of shape (H, W) or (C, H, W).
        mask (torch.Tensor): Segmentation mask with class labels, shape (H, W).
        sign_classes (list[int]): List of class IDs to create overlays for.
        alpha (float, optional): Blending factor for the overlay (0-1). Defaults to 0.55.

    Returns:
        dict[int, torch.Tensor]: A dictionary mapping each class ID to its
            blended overlay image tensor of shape (C, H, W).

    Note:
        Returns an empty dict if sign_classes is empty. All tensors are moved
        to the same device as the mask for efficient computation.

    Example:
        >>> img = torch.rand(3, 224, 224).cuda()
        >>> mask = torch.randint(0, 21, (224, 224)).cuda()
        >>> overlays = create_cs_masks_(img, mask, [1, 5, 10])
    """
    if not sign_classes:
        return {}
    
    # Ensure tensors are on the same device (ideally 'cuda')
    device = mask.device
    sc_img = sc_img.to(device)
    mask = mask.to(device)

    # 1. Create a tensor of class values.
    # Shape: (N,) where N is the number of classes.
    classes_tensor = torch.tensor(sign_classes, device=device, dtype=mask.dtype)

    # 2. Create masks for all classes at once using broadcasting.
    # gt is (H, W), classes_tensor is (N,).
    # We reshape to (1, H, W) and (N, 1, 1) to trigger broadcasting.
    # The result `all_pos_class_gt` will have shape (N, H, W).
    all_pos_class_gt = (mask.unsqueeze(0) == classes_tensor.view(-1, 1, 1))

    # 3. Create all diff masks at once. The op is element-wise.
    # Shape: (N, H, W)

    # 4. Prepare for blending. Convert boolean diff masks to float overlays.
    # Shape: (N, H, W)

    # 5. Blend the single source image with the entire stack of overlays at once.
    # sc_img (H, W) is unsqueezed to (1, H, W) to broadcast across the N overlays.
    # Shape: (N, H, W)
    all_blended_masks = blend_tensors(sc_img.unsqueeze(0), all_pos_class_gt.unsqueeze(1).repeat(1, 3, 1, 1)*255, alpha)

    # 6. Create the final dictionary.
    # This is a fast CPU operation. We pair each class ID with its corresponding mask.
    cs_masks = {
        class_val: mask
        for class_val, mask in zip(sign_classes, all_blended_masks)
    }

    return cs_masks

def create_cs_ovr_masks(
    sc_imgs: torch.Tensor,
    masks: torch.Tensor,
    batch_sign_classes: list[list[int]],
    alpha: float = 0.55
) -> list[dict[int, torch.Tensor]]:
    """
    Creates class-specific overlay masks for a batch of images in a vectorized manner.

    Batch-processes multiple images to create class-specific overlay visualizations.
    For each image in the batch and each of its specified classes, generates a
    blended overlay showing pixels belonging to that class. All operations are
    highly vectorized for GPU efficiency.

    Args:
        sc_imgs (torch.Tensor): Batch of source images. Shape: (B, C, H, W).
        masks (torch.Tensor): Batch of segmentation masks. Shape: (B, H, W).
            Each pixel value represents a class ID.
        batch_sign_classes (list[list[int]]): A list of lists, where each inner list
            contains the class IDs to visualize for the corresponding image in the batch.
            Length must equal batch size B.
        alpha (float, optional): The blending factor for the overlay (0-1).
            Defaults to 0.55.

    Returns:
        list[dict[int, torch.Tensor]]: A list of dictionaries, one per image in the batch.
            Each dictionary maps a class ID to its blended overlay tensor of shape (C, H, W).
            Returns a list of empty dicts if batch_sign_classes is empty.

    Note:
        Uses advanced indexing and broadcasting for efficient batch processing.
        All images are processed simultaneously on the GPU.

    Example:
        >>> imgs = torch.rand(4, 3, 224, 224).cuda()
        >>> masks = torch.randint(0, 21, (4, 224, 224)).cuda()
        >>> classes = [[1, 5], [2], [1, 3, 5], [10]]
        >>> overlays = create_cs_ovr_masks(imgs, masks, classes, alpha=0.5)
        >>> # overlays[0] contains overlays for classes 1 and 5 of the first image
    """
    # 0. Handle empty input
    if not batch_sign_classes:
        return []
    
    batch_size = sc_imgs.shape[0]
    device = masks.device
    sc_imgs = sc_imgs.to(device)

    # 1. Flatten the ragged list of classes into a "long" format.
    # This is a fast, one-time CPU operation.
    flat_classes = []
    batch_indices = []
    for i, class_list in enumerate(batch_sign_classes):
        for class_val in class_list:
            flat_classes.append(class_val)
            batch_indices.append(i)

    # If there are no classes to process across the entire batch, return empty dicts.
    if not flat_classes:
        return [{} for _ in range(batch_size)]

    # Convert to tensors and move to the target device.
    # `flat_classes_tensor` holds all class IDs we need to generate masks for.
    # `batch_indices_tensor` maps each class ID back to its original image in the batch.
    flat_classes_tensor = torch.tensor(flat_classes, device=device, dtype=masks.dtype)
    batch_indices_tensor = torch.tensor(batch_indices, device=device, dtype=torch.long)
    
    # K is the total number of masks to generate across the whole batch.
    K = len(flat_classes_tensor)

    # 2. Use advanced indexing to gather the corresponding masks for each class.
    # `masks` is (B, H, W). We select based on `batch_indices_tensor` (K,).
    # Result `gathered_masks` is (K, H, W).
    gathered_masks = masks[batch_indices_tensor]

    # 3. Create all boolean masks at once using broadcasting.
    # Compare each mask in `gathered_masks` with its corresponding target class value.
    # `gathered_masks` is (K, H, W), `flat_classes_tensor.view(-1, 1, 1)` is (K, 1, 1).
    # Result `all_pos_class_gt` is a boolean tensor of shape (K, H, W).
    all_pos_class_gt = (gathered_masks == flat_classes_tensor.view(-1, 1, 1))

    # 4. Gather the corresponding source images for blending.
    # `sc_imgs` is (B, C, H, W). We select based on `batch_indices_tensor` (K,).
    # Result `gathered_sc_imgs` is (K, C, H, W).
    gathered_sc_imgs = sc_imgs[batch_indices_tensor]

    # 5. Prepare the overlays for blending.
    # Convert boolean masks to float, unsqueeze to add a channel dim, and expand
    # to match the number of channels in the source images.
    # `all_pos_class_gt.unsqueeze(1)` becomes (K, 1, H, W).
    # `expand_as` is more memory-efficient than `repeat`.
    overlays = all_pos_class_gt.unsqueeze(1).expand_as(gathered_sc_imgs) * 255.0

    # 6. Blend all images with their overlays in a single operation.
    # The `blend_tensors` function is vectorized and handles the (K, C, H, W) tensors.
    all_blended_masks = blend_tensors(gathered_sc_imgs, overlays, alpha)

    # 7. Reconstruct the final list of dictionaries.
    # This loop is on the CPU and is very fast as all GPU work is done.
    output_list = [{} for _ in range(batch_size)]
    for i in range(K):
        original_batch_idx = batch_indices[i]
        class_val = flat_classes[i]
        # The i-th blended mask corresponds to the i-th entry in our flat lists.
        output_list[original_batch_idx][class_val] = all_blended_masks[i]

    return output_list

def compute_maps_with_captions(
        vle: VLEncoder,
        images: torch.Tensor,
        captions: list[str],
        map_compute_mode: MapComputeMode,
        map_alpha: int,
        viz_image_size: Optional[int | list[int]] = None,
        map_resize_mode: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
        normalize: bool = True,
        with_patch_text: bool = False,
        font_path: Path = Path('/home/olivieri/exp/resources/Arial.ttf'),
        font_size: int = 24,
) -> list[tuple[Image.Image, str]]:
    image_text_list = []

    for img, text in zip(images, captions):
        img_tensor = vle.preprocess_images([img])
        text_tensor = vle.preprocess_texts([text])
        img_output = vle.encode_and_project_images(img_tensor)
        txt_output = vle.encode_and_project_texts(text_tensor)
        sim = vle.get_similarity(img_output, txt_output, broadcast=False)
        map, indices = vle.get_maps(
            img_output,
            txt_output,
            map_compute_mode=map_compute_mode,
            upsample_size=viz_image_size,
            upsample_mode=map_resize_mode,
            broadcast=False
        ) # [1, H, W]
        patches_dims = (np.array(img_tensor.shape[-2:])//np.array(vle.patch_size)).astype(int).tolist()
        if indices is not None:
            indices = indices.view(indices.shape[0], *patches_dims)
        if viz_image_size:
            img = TF.resize(img, size=viz_image_size, interpolation=TF.InterpolationMode.BILINEAR)
        ovr_img = overlay_map(img, map, normalize=normalize, alpha=map_alpha) # (H_viz, W_viz)
        viz_patch_size = (np.array([vle.patch_size])*(np.array(viz_image_size))/np.array(img_tensor.shape[-2:])).astype(int).tolist()[0]
        if with_patch_text and (indices is not None):
            texts_np = text_tensor[0, indices[0]].cpu().numpy()
            texts_np = vle.decode_tokens(texts_np)
            ovr_img = draw_text_in_patches_center(ovr_img, patch_size=viz_patch_size, texts_np=texts_np, font_path=font_path, font_size=font_size, text_color='black')

        image_text_list.append((ovr_img, f"SIM = {sim.item():.2f}", f"MIN VALUE = {map.min().item():.2f}, MAX VALUE = {map.max().item():.2f}", text, "---"))

    return image_text_list

def compute_cs_maps_with_captions(
        viz_ds: VOC2012SegDataset,
        img_idxs: list[int],
        jsonl_path: Path,
        vle: VLEncoder,
        mask_color: Literal['L', 'RB'],
        alpha: float,
        map_alpha: float,
        map_compute_mode: MapComputeMode,
        map_resize_mode: TF.InterpolationMode,
        viz_image_size: int | tuple[int, int],
        normalize: bool,
        contrastive: bool = False,
        with_patch_text: bool = False,
        font_path: Path = Path('/home/olivieri/exp/resources/Arial.ttf'),
        font_size: int = 24,
) -> list[str | Image.Image]:
    cs_answers = get_answer_objects(jsonl_path, idxs=None, jsonlio=JsonlIO(), return_state=False, format_to_dict=True)
    cs_answers_text = [list(cs_answers[i].values()) for i in img_idxs]
    
    scs, gts, prs = viz_ds[:]

    cs_ovr_mask_prs = []

    for sc, gt, pr in zip(scs, gts, prs):
        gt_sign_classes = get_significant_classes(gt)
        pr_sign_classes = get_significant_classes(pr)
        sign_classes = list(set(gt_sign_classes + pr_sign_classes))
        if 0 in sign_classes and sign_classes != [0]:
            sign_classes.remove(0)
        sign_classes = sorted(sign_classes)

        ovr_mask_prs = []

        for pos_c in sign_classes:
            pos_class_gt = (gt == pos_c)
            pos_class_pr = (pr == pos_c)

            diff_mask = create_diff_mask(pos_class_gt, pos_class_pr)

            # L overlay image
            ovr_diff_mask_L = blend_tensors(sc, diff_mask*255, alpha)

            # RB overlay image
            diff_mask += (diff_mask*pos_class_gt) # sets to 2 the false negatives
            diff_mask_col_RB = apply_colormap([diff_mask], {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 0, 255)}).squeeze()
            ovr_diff_mask_RB = blend_tensors(sc, diff_mask_col_RB, alpha)

            if mask_color == 'L':
                ovr_mask_prs.append(ovr_diff_mask_L)
            elif mask_color == 'RB':
                ovr_mask_prs.append(ovr_diff_mask_RB)

        if contrastive:
            ovr_mask_prs = [ovr_mask_prs[0]]*int(len(ovr_mask_prs)) # instead of show consider positive images, show only the first one.

        cs_ovr_mask_prs.append(ovr_mask_prs)
    
    cs_pr_image_text_list = [
        compute_maps_with_captions(
            vle=vle,
            images=ovr_mask_prs,
            captions=answers_pr_text,
            map_compute_mode=map_compute_mode,
            map_alpha=map_alpha,
            viz_image_size=viz_image_size,
            map_resize_mode=map_resize_mode,
            normalize=normalize,
            with_patch_text=with_patch_text,
            font_path=font_path,
            font_size=font_size,
        )
        for ovr_mask_prs, answers_pr_text in zip(cs_ovr_mask_prs, cs_answers_text)
    ]
    
    return cs_pr_image_text_list

@torch.inference_mode()
def compute_cs_gt_pr_maps_with_captions(
        viz_ds: VOC2012SegDataset,
        img_idxs: Optional[list[int] | slice],
        jsonl_path: Path,
        vle: VLEncoder,
        mask_color: Literal['L', 'RB'],
        mask_alpha: float,
        map_alpha: float,
        map_compute_mode: MapComputeMode,
        map_resize_mode: TF.InterpolationMode,
        viz_image_size: int | tuple[int, int],
        normalize: bool,
        with_patch_text: bool = False,
        font_path: Path = Path('/home/olivieri/exp/resources/Arial.ttf'),
        font_size: int = 24,
) -> list[str | Image.Image]:
    cs_answers_dict: dict[int, dict[int, str]] = get_answer_objects(jsonl_path, idxs=img_idxs, jsonlio=JsonlIO(), return_state=False, format_to_dict=True)

    cs_dict = {uid: list(pos_classes.keys()) for uid, pos_classes in cs_answers_dict.items()}

    uids = list(cs_dict.keys())
    pos_classes = list(cs_dict.values())
    flat_pos_classes, _ = flatten_list_of_lists(pos_classes)
    flat_pos_classes = [int(pos_c) for pos_c in flat_pos_classes]

    cs_texts = [list(cs_a.values()) for cs_a in cs_answers_dict.values()]
    flat_cs_texts, lengths = flatten_list_of_lists(cs_texts)
    flat_cs_texts_t = vle.preprocess_texts(flat_cs_texts) # list of class-splitted tensors (P, tokens)

    P = len(flat_cs_texts_t)
    
    scs_img, gts, prs = viz_ds[img_idxs]

    scs_img = torch.stack(scs_img) # (B, 3, H, W)
    gts = torch.stack(gts) # (B, 1, H, W)
    prs = torch.stack(prs) # (B, 1, H, W)

    unrolled_scs_img = torch.cat([sc.unsqueeze(0).expand(len(cs_dict[uid]), -1, -1, -1) for sc, uid in zip(scs_img, uids)]) # (P, 3, H, W)
    unrolled_gts = torch.cat([gt.unsqueeze(0).expand(len(cs_dict[uid]), -1, -1, -1) for gt, uid in zip(gts, uids)]) # (P, 1, H, W)
    unrolled_prs = torch.cat([pr.unsqueeze(0).expand(len(cs_dict[uid]), -1, -1, -1) for pr, uid in zip(prs, uids)]) # (P, 1, H, W)

    concat_masks = torch.cat([unrolled_gts, unrolled_prs], dim=0) # (2P, H, W)

    ovr_concat_masks_img = format_pos_ovr_masks(unrolled_scs_img, concat_masks, flat_pos_classes, mask_alpha) # (2P, 3, H, W)

    ovr_concat_masks = vle.preprocess_images(ovr_concat_masks_img) # (2P, 3, H, W)

    img_output = vle.encode_and_project_images(ovr_concat_masks) # (2P, D)
    txt_output = vle.encode_and_project_texts(flat_cs_texts_t) # # (P, D)

    # NOTE the following computation can be further sped up by splitting in the end, after the concat_adapter

    gt_global_image_token = img_output.global_image_token[:P] # (P, D)
    gt_local_image_tokens = img_output.local_image_tokens[:P] # (P, D)

    pr_global_image_token = img_output.global_image_token[P:] # (P, D)
    pr_local_image_tokens = img_output.local_image_tokens[P:] # (P, D)

    lhs_global: torch.Tensor = vle.model.concat_adapter(torch.cat([gt_global_image_token, pr_global_image_token], dim=-1)) # (P, D)
    lhs_local: torch.Tensor = vle.model.concat_adapter(torch.cat([gt_local_image_tokens, pr_local_image_tokens], dim=-1)) # (P, D)

    new_img_output = VLEImageOutput(lhs_global, lhs_local)
    new_txt_output = VLETextOutput(txt_output.global_text_token, txt_output.local_text_tokens)

    sims = vle.get_similarity(new_img_output, new_txt_output, broadcast=False)

    maps, indices = vle.get_maps(
        new_img_output,
        new_txt_output,
        map_compute_mode=map_compute_mode,
        upsample_size=viz_image_size,
        upsample_mode=map_resize_mode,
        broadcast=False,
    )

    patches_dims = (np.array(ovr_concat_masks.shape[-2:])//np.array(vle.patch_size)).astype(int).tolist()
    if indices is not None:
        indices = indices.view(indices.shape[0], *patches_dims)

    cs_pr_image_text_list = []

    for i, map in enumerate(maps):
        # RB overlay image
        sc = unrolled_scs_img[i]
        pos_c = flat_pos_classes[i]
        gt = (unrolled_gts[i] == pos_c)
        pr = (unrolled_prs[i] == pos_c)
        diff_mask = create_diff_mask(gt, pr)
        diff_mask += (diff_mask*gt) # sets to 2 the false negatives
        if mask_color == 'L':
            ovr_img = blend_tensors(sc, diff_mask*255, mask_alpha)
        elif mask_color == 'RB':
            diff_mask_col_RB = apply_colormap([diff_mask], {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 0, 255)}).squeeze()
            ovr_img = blend_tensors(sc, diff_mask_col_RB, mask_alpha)
        if viz_image_size:
            ovr_img = TF.resize(ovr_img, size=viz_image_size, interpolation=TF.InterpolationMode.BILINEAR)
        
        ovr_img = overlay_map(ovr_img, map.unsqueeze(0), normalize=normalize, alpha=map_alpha) # (H_viz, W_viz)

        viz_patch_size = (np.array([vle.patch_size])*(np.array(viz_image_size))/np.array(ovr_concat_masks.shape[-2:])).astype(int).tolist()[0]

        if with_patch_text and (indices is not None):
            # texts_np=np.array(["BACKGROUND"]*196).reshape(14, 14)
            # texts_np = indices[i].cpu().numpy()
            # texts_np = np.vectorize(lambda i: str(i))(texts_np)
            texts_np = flat_cs_texts_t[i, indices[i]].cpu().numpy()
            texts_np = vle.decode_tokens(texts_np)
            ovr_img = draw_text_in_patches_center(ovr_img, patch_size=viz_patch_size, texts_np=texts_np, font_path=font_path, font_size=font_size, text_color='black')

        sim = sims[i]

        text = flat_cs_texts[i]

        cs_pr_image_text_list.append((ovr_img, f"SIM = {sim.item():.2f}", f"MIN VALUE = {map.min().item():.2f}, MAX VALUE = {map.max().item():.2f}", text, "---"))

    # TODO implement contrastive

    return cs_pr_image_text_list

def draw_text_in_patches_center(
        image: Image.Image,
        patch_size: tuple[int, int],
        texts_np: np.ndarray[str],
        font_path: Path,
        font_size: int = 24,
        text_color: str = "black"
) -> Image.Image:
    """
    Draws a specific text string centered in each conceptual patch of an image.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (tuple): A tuple (width, height) for the patch size.
        texts (list of lists): A 2D list of strings. Its dimensions MUST match the
                               number of patches (rows x columns).
        font_size (int): The desired font size.
        font_path (str): Path to the .ttf or .otf font file.
        text_color (str or tuple): The color of the text.
        
    Raises:
        ValueError: If the dimensions of the `texts` grid do not match the
                    number of patches calculated from the image and patch size.
    """
    patch_width, patch_height = patch_size
    img_width, img_height = image.size

    # Calculate the number of patches in each dimension
    num_cols = img_width // patch_width
    num_rows = img_height // patch_height

    # --- VALIDATION BLOCK ---
    # 1. Check if the number of rows in the text grid matches
    if len(texts_np) != num_rows:
        raise ValueError(
            f"The number of rows in 'texts' ({len(texts_np)}) does not match the "
            f"calculated number of patch rows ({num_rows})."
        )
    # 2. Check if the number of columns in each row of the text grid matches
    for i, row_of_texts in enumerate(texts_np):
        if len(row_of_texts) != num_cols:
            raise ValueError(
                f"The number of columns in row {i} of 'texts' ({len(row_of_texts)}) "
                f"does not match the calculated number of patch columns ({num_cols})."
            )
    # --- END VALIDATION BLOCK ---

    # Create a drawing context
    draw = PIL.ImageDraw.Draw(image)
    
    # Load the font
    try:
        font = PIL.ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Font not found at '{font_path}'. Using default font.")
        font = PIL.ImageFont.load_default()

    # Iterate over each patch
    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate the center of the patch
            center_x = (col * patch_width) + (patch_width / 2)
            center_y = (row * patch_height) + (patch_height / 2)

            # Get the text for the current patch
            text = texts_np[row][col]

            # Draw the text, centered on the coordinate
            draw.text(
                (center_x, center_y),
                text,
                font=font,
                fill=text_color,
                anchor="mm"  # Requires Pillow 10.0.0+
            )

    return image

def viz_seg_saliency_maps(
        sc_imgs: torch.Tensor, # (B, 3, H, W)
        maps: torch.Tensor, # (B, H, W)
        flat_target_classes: list[int],
        flat_target_class_names: list[str],
        gt_masks: torch.Tensor,
        pr_masks: torch.Tensor,
        normalise: bool = True,
        alpha: int = 0.5,
        viz_img_size: Optional[int | tuple[int, int]] = None
) -> Figure:
    if normalise:
        maps = normalize_sim_maps(maps)

    maps = F.interpolate(
            maps.unsqueeze(1), # (B, 1, H, W)
            size=sc_imgs.shape[-2:],
            mode='bilinear',
    ).squeeze(1) # (B, H_viz, W_viz)
    # scs_imgs = F.interpolate(scs_imgs, size=maps.shape[-2:], mode='bilinear') # (B, 3, H_viz, W_viz)
    
    # format gt and pr images with the selected classes
    gt_imgs = (gt_masks == torch.tensor(flat_target_classes, device=gt_masks.device).view(-1, 1, 1)).unsqueeze(1).expand(-1, 3, -1, -1)*1.0
    pr_imgs = (pr_masks == torch.tensor(flat_target_classes, device=pr_masks.device).view(-1, 1, 1)).unsqueeze(1).expand(-1, 3, -1, -1)*1.0

    gt = gt_masks == torch.tensor(flat_target_classes, device=gt_masks.device).view(-1, 1, 1)
    pr = pr_masks == torch.tensor(flat_target_classes, device=gt_masks.device).view(-1, 1, 1)

    intersection = (gt & pr).sum(dim=(1,2)).float()
    union = (gt | pr).sum(dim=(1,2)).float()

    m_ious = intersection / union.clamp(min=1e-6) # (B)

    # Create figure with subplots
    fig, axes = plt.subplots(len(sc_imgs), 4, figsize=(20, 5*len(sc_imgs)))
    
    # Select the first image in the batch
    for i in range(len(sc_imgs)):
        ax = axes[i]

        sc_img = sc_imgs[i].permute(1, 2, 0).cpu().numpy()  # [H_viz, W_viz, C]
        gt_img = gt_imgs[i].permute(1, 2, 0).cpu().numpy()  # [H_viz, W_viz, C]
        pr_img = pr_imgs[i].permute(1, 2, 0).cpu().numpy()  # [H_viz, W_viz, C]
        cam_heatmap = maps[i].detach().cpu().numpy()  # [H_viz, W_viz]

        # 1. SC
        ax_ = ax[0]
        ax_.imshow(sc_img)
        ax_.set_title(f'SC (index {i})')
        ax_.axis('off')

        # convert heatmap to RGB using jet colormap
        heatmap_colored = cm.jet(cam_heatmap)[:, :, :3] # remove alpha channel
        
        # 2. MAP
        ax_ = ax[1]
        ax_.imshow(heatmap_colored)
        ax_.set_title(f'MAP (class {flat_target_classes[i]} - {flat_target_class_names[i]})')
        ax_.axis('off')

        # 3. GT + MAP
        ax_ = ax[2]
        ax_.imshow(gt_img)
        ax_.set_title('GT')
        ax_.axis('off')

        # 4. PR + MAP
        ax_ = ax[3]
        ax_.imshow(pr_img)
        ax_.set_title(f'PR - mIoU={m_ious[i]:0.3}')
        ax_.axis('off')

    return fig
