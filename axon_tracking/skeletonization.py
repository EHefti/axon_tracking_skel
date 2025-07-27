import os

import cloudvolume as cv
import kimimaro
import numpy as np
import scipy.ndimage as nd
import scipy.stats as stats
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from skimage.feature import peak_local_max
from skimage.morphology import ball
from scipy.ndimage import binary_dilation, generate_binary_structure
import matplotlib.pyplot as plt

from axon_tracking import utils as ut


def full_skeletonization(root_path, template_id, params, skel_params, qc_params):
    template, template_save_file, noise = load_template_file(root_path, template_id)
    if np.mean(noise) > params["max_noise_level"]:
        return [], []

    temp_diff = np.diff(template)
    capped_template, target_coor = localize_neurons(
        temp_diff, ms_cutout=params["ms_cutout"]
    )
    tmp_filt = nd.gaussian_filter(capped_template, sigma=1)
    interp_temp = interpolate_template(tmp_filt, spacing=params["upsample"])
    th_template = threshold_template(interp_temp, noise, target_coor, params)

    t_cap = [0, th_template.shape[2]]  # in samples
    skels = skeletonize(
        th_template[:, :, t_cap[0] : t_cap[1]].astype("bool"), **skel_params
    )
    skeleton = kimimaro.join_close_components(skels[1], radius=5)
    skeleton = kimimaro.postprocess(skeleton, tick_threshold=5, dust_threshold=10)

    all_branches = branches_from_paths(skeleton)
    scaled_qc_list, r2s, vels, lengths = perform_path_qc(
        all_branches, params, **qc_params
    )

    qc_skel_list = [cv.Skeleton.from_path(x) for x in scaled_qc_list]
    qc_skeleton = cv.Skeleton.simple_merge(qc_skel_list)
    qc_skeleton = kimimaro.postprocess(
        qc_skeleton, dust_threshold=10, tick_threshold=10
    )
    qc_skeleton = kimimaro.join_close_components(qc_skeleton, radius=200)
    qc_skeleton = kimimaro.postprocess(qc_skeleton, dust_threshold=0, tick_threshold=0)

    all_branches = branches_from_paths(qc_skeleton)

    scaled_qc_list, r2s, full_vels, lengths = perform_path_qc(
        all_branches, params, **qc_params
    )

    return qc_skeleton, scaled_qc_list


def load_template_file(root_path, template_id):
    """
    Load a template file.
    This function loads a template file from the specified root path and template ID.
    The template file is expected to be in NumPy (.npy) format.
    Args:
        root_path (str): The root directory path where the template file is located.
        template_id (int): The identifier for the template file. The file is expected
                            to be named as '<template_id>.npy'.
    Returns:
        tuple: A tuple containing:
            - template (numpy.ndarray): The loaded template as a NumPy array of type float64.
            - template_save_file (str): The full path to the loaded template file.
    """

    template_save_file = os.path.join(root_path, str(template_id) + ".npy")
    template = np.load(template_save_file).astype("float64")

    return template, template_save_file


def preprocess_template(template, params):
    """
    Preprocess a template by localizing the AIS and thresholding the template.
    This function preprocesses a given template by localizing the AIS and thresholding
    the template based on the absolute signal amplitude, the maximum velocity, and the
    noise of the electrode.
    Args:
        template (numpy.ndarray): The template as a NumPy array.
        params (dict): A dictionary containing the parameters for preprocessing.
    Returns:
        numpy.ndarray: The interpolated template.
        numpy.ndarray: The noise matrix.
        numpy.ndarray: The thresholded (boolean) template.
    """
    # Calculate the noise matrix based on the template
    noise = generate_noise_matrix(template, mode="mad")
    # Large median filter to remove noise
    med_filt = nd.median_filter(template, footprint=ball(2))
    # First derivative to eliminate slow drift
    temp_diff = np.diff(med_filt)
    # Median filter to remove noise from the derivative
    tmp_filt = nd.median_filter(temp_diff, footprint=ball(1))

    # Interpolate the template in x, y, and z (time)
    interp_temp = interpolate_template(tmp_filt, spacing=params["upsample"])

    # Localize neurons based on the derivative
    if params["ais_detection"] is not None:
        interp_temp, ais = localize_ais(interp_temp, params)
    else:
        ais = np.array([])

    # Generate a noise matrix based on the template
    interp_noise = interpolate_template(noise, spacing=params["upsample"])

    return interp_temp, interp_noise, ais


def localize_ais(input_mat, params):
    """Localize the AIS in a given template.
    This function localizes AIS in a given input matrix using the `peak_local_max`
    function from the `skimage` library to detect the peaks.
    Args:
        input_mat (numpy.ndarray): The input matrix as a NumPy array.
        params (dict): A dictionary containing the parameters for localization.
        min_distance (int): The minimum distance between peaks.
        threshold_rel (float): The relative threshold for peaks.
    Returns:
        numpy.ndarray: The capped matrix containing the localized neurons.
        numpy.ndarray: The coordinates of the AIS peak.
    """
    peak_idx = params["ms_cutout"][0] * params["sampling_rate"] / 1000
    peak_cutout = (
        params["upsample"][2] * params["sampling_rate"] / 1000
    )  # Cutout in ms before and after the peak
    local_max = peak_local_max(
        np.abs(input_mat), min_distance=3, threshold_rel=0.1, num_peaks=10
    )

    # Safety check for no peaks
    if local_max.size == 0:
        return input_mat, np.array([])

    if (
        params["ais_detection"] == "dev"
    ):  # Search for the minimum deviation from expected peak time
        ais = local_max[np.argmin(local_max[:, 2] - peak_idx), :]
    elif params["ais_detection"] == "time":  # Search for first peak
        ais = local_max[np.argmin(local_max[:, 2]), :]
    elif params["ais_detection"] == "amp":  # Search for highest peak
        ais = local_max[0, :]
    else:
        raise ValueError("Invalid search mode")

    # Cap the matrix at the AIS peak time and add a buffer
    ais_buffer = np.min([ais[2], params["buffer_frames"]])  # Prevent negative indices
    ais_peak = int(ais[2] - ais_buffer)
    capped_matrix = input_mat[:, :, ais_peak:]
    ais[2] = ais_buffer
    return capped_matrix, ais


def generate_noise_matrix(template, mode="mad"):
    """Generate a noise matrix based on the template.
    This function generates a noise matrix based on the template. The noise matrix is
    calculated as the median absolute deviation (MAD) or standard deviation (SD) of the
    template along the third axis.
    Args:
        template (numpy.ndarray): The template as a NumPy array
        noise (numpy.ndarray): The noise matrix to use. If not provided, the noise matrix
                                 is generated based on the template.
        mode (str): The mode to use for noise calculation. Can be either 'mad' for median
                    absolute deviation or 'sd' for standard deviation.
    Returns:
        numpy.ndarray: The generated noise matrix.
    """

    if mode == "mad":
        noise = stats.median_abs_deviation(template, axis=2)
    elif mode == "sd":
        noise = np.std(template, axis=2)

    noise_matrix = noise[:, :, np.newaxis]  # For compatibility with the template shape

    return noise_matrix


def threshold_template(template, noise, target_coor, params):
    """Generate a thresholded template based on the absolute signal amplitude, the
    maximum velocity, and the noise of the electrode (taken from noise matrix).
    Args:
        template (numpy.ndarray): The template as a NumPy array.
        noise (numpy.ndarray): The noise matrix as a NumPy array.
        target_coor (list): The target coordinates for the template.
        params (dict): A dictionary containing the parameters for thresholding.
    Returns:
        numpy.ndarray: The boolean thresholded template.
    """

    # mad_noise = generate_noise_matrix(template, mode="mad")
    # sd_noise = generate_noise_matrix(template, mode="sd")
    if params["noise_threshold"] is not None:
        noise_th = template < (
            params["noise_threshold"] * noise[:, :, : template.shape[2]]
        )
        # print("Noise thresholding")
    else:
        noise_th = np.full_like(template, True)
    abs_th = template < params["abs_threshold"]
    velocity_th = valid_latency_map(template, target_coor, params)
    th_template = noise_th * abs_th * velocity_th  # * ((sd_noise / mad_noise) > 1)
    return th_template


def valid_latency_map(template, start, params):
    """Generate a boolean matrix indicating the valid latency map based on the maximum
    velocity.
    Args:
        template (numpy.ndarray): The template as a NumPy array.
        start (list): The starting coordinates (AIS) for velocity calculations.
        params (dict): A dictionary containing the parameters for thresholding.
    Returns:
        numpy.ndarray: The boolean matrix indicating the valid latency map.
    """

    indices_array = np.indices(template.shape) * params["el_spacing"]  # convert to (um)
    # Calculate the distance from the start point
    distances = (
        np.sqrt(
            (indices_array[0] - start[0] * params["el_spacing"]) ** 2
            + (indices_array[1] - start[1] * params["el_spacing"]) ** 2
        )
        / 1000000
    )  # distances in m
    delay_mat = np.zeros(distances.shape)  # Initialize delay matrix
    for z in range(distances.shape[2]):
        # Calculate the delay matrix based on the distance from the start point in
        # the z-axis
        delay_mat[:, :, z] = (1 / params["sampling_rate"]) * np.abs((z - start[2]))

    # Set the delay matrix for the starting point to non-zero
    delay_mat[:, :, start[2]] = 1 / params["sampling_rate"]

    # Check which velocity is within the maximum velocity
    passed = (distances / delay_mat) < params["max_velocity"]  # Boolean matrix
    return passed


def restore_sparse_template(template, spacing=(0.5, 0.5, 0.5)):
    """Remove empty electrodes and interpolate the sparse template to fill in the gaps.
    Args:
        template (numpy.ndarray): The template as a NumPy array.
        spacing (tuple): The spacing for interpolation.
    Returns:
        numpy.ndarray: The interpolated template.
    """

    # Find where the actual template begins (first non-zero electrode)
    x, y, z = np.nonzero(template)
    # Restrict template to the smallest bounding box
    true_template = template[np.min(x) :, np.min(x) :, :]
    # Remove empty electrodes (due to sparseness)
    del_x = np.delete(
        true_template, np.nonzero(np.sum(sel_test, axis=(0, 2)) == 0), axis=1
    )
    del_y = np.delete(del_x, np.nonzero(np.sum(sel_test, axis=(1, 2)) == 0), axis=0)
    # Interpolate the template to fill in the gaps
    interp_tmp = interpolate_template(del_y, spacing)
    return interp_tmp


def interp_max(x, spacing):
    """Calculate the maximum value for interpolation.
    Args:
        x (numpy.ndarray): The input array.
        spacing (int): The spacing for interpolation.
    Returns:
        int: The maximum value for interpolation.
    """

    if len(x) == 1:
        interp_max = spacing
    elif spacing == 1:
        interp_max = x[-1] + 1
    else:
        interp_max = x[-1]
    return interp_max


def interpolate_template(
    template, spacing=[1, 1, 0.2], template_path=[], overwrite=False
):
    """Generate an interpolated template based on the input template and spacing.
    Args:
        template (numpy.ndarray): The template as a NumPy array.
        spacing (list): The spacing for interpolation.
        template_path (str): The path to save the interpolated file to (optional).
        overwrite (bool): Whether to overwrite the existing interpolated file.
    Returns:
        numpy.ndarray: The interpolated template.
    """

    if template_path:
        split_path = template_path.split(sep="/")
        split_path[-1] = "interp_" + split_path[-1]
        interp_tmp_path = "/".join(split_path)
    else:
        interp_tmp_path = ""

    if os.path.exists(interp_tmp_path) and not overwrite:
        interp_template = np.load(interp_tmp_path).astype("float64")

    else:
        x, y, z = [np.arange(template.shape[k]) for k in range(3)]
        f = RegularGridInterpolator((x, y, z), template)
        new_grid = np.mgrid[
            0 : interp_max(x, spacing[0]) : spacing[0],
            0 : interp_max(y, spacing[1]) : spacing[1],
            0 : interp_max(z, spacing[2]) : spacing[2],
        ]

        new_grid = np.moveaxis(
            new_grid, (0, 1, 2, 3), (3, 0, 1, 2)
        )  # reorder axes for evaluation
        interp_template = f(new_grid)

        if template_path:
            np.save(interp_tmp_path, interp_template.astype("float32"))

    return interp_template


def cone(matrix_shape, r, apex=[]):
    if not apex:
        apex = np.empty(3)
        apex[0], apex[1], apex[2] = (
            np.floor(matrix_shape[0] / 2),
            np.floor(matrix_shape[1] / 2),
            0,
        )

    x, y, z = np.ogrid[: matrix_shape[0], : matrix_shape[1], : matrix_shape[2]]
    cone_equation = (x - apex[0]) ** 2 + (y - apex[1]) ** 2 <= r**2 * (
        1 - (z - apex[2]) / matrix_shape[2]
    ) ** 2
    cone_matrix = np.zeros(matrix_shape, dtype=bool)
    cone_matrix[cone_equation] = True
    return cone_matrix


def generate_dilation_structure(max_t, params):
    """Generate a cone-shaped dilation structure based on the maximum time and parameters.
    Args:
        max_t (int): The maximum time.
        params (dict): A dictionary containing the parameters for dilation.
    Returns:
        numpy.ndarray: The generated dilation structure.

    """
    frame_time = 1000000 / params["sampling_rate"]  # in us
    t = np.ceil(max_t / frame_time).astype("int16")  # in samples
    max_r = (
        params["max_velocity"] * max_t
    )  # Infer the maximum radius from the maximum velocity
    r = np.ceil(max_r / params["el_spacing"]).astype("int16")  # in grid units
    d = (2 * r + 1).astype("int16")  # maximum diameter of the cone

    cone_matrix = cone((d, d, t), r)  # Generate the cone matrix

    structure = cone_matrix[:, :, ::-1]
    structure_base = np.full(
        (structure.shape[0], structure.shape[1], structure.shape[2]), False
    )
    structure_init = np.full((structure.shape[0], structure.shape[1], 1), False)
    structure_init[r, r, 0] = True
    full_structure = np.concatenate(
        (structure_base, structure_init, structure), axis=2
    )  # Ensure that dilation only occurs in the future

    return full_structure


def iterative_dilation(template, params):

    structure = generate_dilation_structure(r_dilation)
    m_init = template < init_th  # Detection of initial seeds/definitive peaks
    mask = template < min_th  # Mask indicating potential peak locations
    dilated = nd.binary_dilation(m_init, structure=structure, iterations=0, mask=mask)
    # filtered = nd.median_filter(dilated,size=filter_size)
    # filtered = nd.median_filter(dilated,footprint=ball(1))
    if filter_footprint is not None:
        dilated = nd.median_filter(dilated, footprint=filter_footprint)

    return dilated


def skeletonize(
    input_matrix,
    scale=2,
    const=50,
    pdrf_exponent=4,
    pdrf_scale=10000,
    dust_threshold=0,
    anisotropy=(17.5, 17.5, 50.0),
    tick_threshold=10,
    n_jobs=16,
    ais=[],
):
    skels = kimimaro.skeletonize(
        input_matrix,
        teasar_params={
            "scale": scale,
            "const": const,  # physical units
            "pdrf_exponent": pdrf_exponent,
            "pdrf_scale": pdrf_scale,
            "soma_acceptance_threshold": 35,  # physical units
            "soma_detection_threshold": 20,  # physical units
            "soma_invalidation_const": 30,  # physical units
            "soma_invalidation_scale": 2,
        },
        dust_threshold=dust_threshold,
        parallel=n_jobs,
    )
    # extra_targets_before=ais

    return skels


def perform_path_qc(
    paths,
    params,
    window_size=7,
    max_duplicate_ratio=0.5,
    min_r2=0.9,
    vel_range=[0.4, 1],
    min_length=10,
):

    scaled_paths = ut.convert_coor_scale(paths, params, scale="um")
    good_path_list, r2s, vels, lengths = [], [], [], []
    path_list = []
    for path in scaled_paths:
        if len(path) > window_size:
            path, inflection_points = split_path(
                path, window_size=window_size, max_duplicate_ratio=max_duplicate_ratio
            )
            path = [p for p in path if len(p) > min_length]
        if len(path) < 1:
            continue
        if type(path) is not list:
            path = [path]
        good_path_list = good_path_list + path
    # good_path_list.append(path)
    # print(good_path_list)
    for path in good_path_list:
        if path.shape[0] > 3:
            vel, r2 = calculate_path_velocity(path, params)
        else:
            vel, r2 = np.nan, np.nan
        r2s.append(r2)
        vels.append(vel)
        lengths.append(len(path))

    good_vel = np.where(
        (np.array(r2s) > min_r2)
        & (np.array(vels) < vel_range[1])
        & (np.array(vels) > vel_range[0])
    )
    good_vel = good_vel | np.isnan(r2)
    qc_list = [good_path_list[x] for x in good_vel[0]]
    r2s = [r2s[x] for x in good_vel[0]]
    vels = [vels[x] for x in good_vel[0]]
    lengths = [lengths[x] for x in good_vel[0]]
    return qc_list, r2s, vels, lengths


def split_path(path, window_size=7, max_duplicate_ratio=0.3):
    smoothed_data = np.convolve(
        path[:, 2], np.ones(window_size) / window_size, mode="valid"
    )
    gradient = np.gradient(smoothed_data)
    inflection_points = np.where(np.diff(np.sign(gradient)))[0]
    inflection_points = (inflection_points + np.floor(window_size / 2)).astype("int")
    inflection_points = np.delete(
        inflection_points, [np.where(np.abs(np.diff(inflection_points)) < 2)]
    )
    split_points = [0] + list(inflection_points) + [path.shape[0]]
    # print(split_points)
    split_paths = [
        path[split_points[s] : split_points[s + 1], :]
        for s in range(len(split_points) - 1)
    ]
    qc_paths = remove_circulating_paths(
        split_paths, max_duplicate_ratio=max_duplicate_ratio
    )

    return qc_paths, inflection_points


def branches_from_paths(skeleton):
    branch_points = skeleton.branches()
    path_list = skeleton.paths()
    all_branches = []
    for path in path_list:
        path_idx, _ = np.where(
            (path[:, None] == skeleton.vertices[branch_points]).all(axis=2)
        )
        branch_idx = []
        for p in path_idx:
            # check_idx = range(p-1,p+2)
            dists = pdist(path[p - 1 : p + 2, :])
            branch_idx = [0]
            if dists[0] > dists[2]:
                branch_idx.append(p)
            else:
                branch_idx.append(p + 1)
        if len(branch_idx) > 0:
            branch_idx.append(path.shape[0])
            new_branches = [
                path[branch_idx[x] : branch_idx[x + 1], :]
                for x in range(len(branch_idx) - 1)
            ]
            all_branches = all_branches + new_branches

    unique_branches = []
    for arr in all_branches:
        # Check if the array is not already in unique_arrays
        branch_list = []
        if not any(np.array_equal(arr, unique_arr) for unique_arr in unique_branches):
            # unique_branches.append(arr)
            branch_list.append(arr)
        unique_branches = unique_branches + branch_list
        unique_branches = [x for x in unique_branches if x.shape[0] > 1]
    return unique_branches


def remove_circulating_paths(path_list, max_duplicate_ratio=0.3):
    """
    Removes paths that contain too many duplicate x,y coordinates, expressed as ratio.
    """
    indices_to_remove = []
    for p, path in enumerate(path_list):
        points = path[:, :2]
        duplicate_ratio = 1 - (len(np.unique(points, axis=0)) / len(points))

        if duplicate_ratio > max_duplicate_ratio:
            indices_to_remove.append(p)

    clean_path_list = [
        path_list[i] for i in range(len(path_list)) if i not in indices_to_remove
    ]
    # print(len(path_list) - len(clean_path_list))
    return clean_path_list


def calculate_path_velocity(path, params):
    path_diff = np.diff(path, axis=0)
    # print(path_diff)
    # print(path_diff.shape)
    dist = np.sqrt(path_diff[:, 0] ** 2 + path_diff[:, 1] ** 2) / 1000
    time = (np.cumsum(np.abs(path_diff[:, 2])) / params["sampling_rate"]) * 1000
    regressor = LinearRegression(fit_intercept=True)
    vel_y = np.cumsum(dist).reshape(-1, 1)
    vel_x = time.reshape(-1, 1)
    regressor.fit(vel_x, vel_y)
    y_pred = regressor.predict(vel_x)
    r2 = r2_score(vel_y, y_pred)
    return regressor.coef_[0][0], r2


def path_to_vertices(path_list, params, unscale=True):
    # if unscale:
    #     path_list = unscale_path_coordinates(path_list, params)

    vertices = np.concatenate(path_list)
    sorted_indices = np.argsort(vertices[:, 2])
    sorted_vertices = vertices[sorted_indices, :]
    return sorted_vertices



# ---------------------------------------------- Extra - Removal of peaks on every Template ---------------------------------------------------- #



def identify_ubiquitous_peaks(templates, min_distance=5, threshold_abs=None, threshold_rel=0.1):
    """
    Calculates the median and finds peaks using skimage.feature.peak_local_max.

    Args:
        templates: List of 3D numpy arrays.
        min_distance: Minimum distance between peaks (in voxels).
        threshold_abs: Absolute intensity threshold.
        threshold_rel: Relative intensity threshold (fraction of max value).

    Returns:
        median_template, peak_coordinates (as a NumPy array)
    """
    stacked_templates = np.stack(templates, axis=0)
    median_template = np.median(stacked_templates, axis=0)
    # print(median_template.shape[:2])

    peak_coordinates = peak_local_max(
        median_template,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        exclude_border=False
    )
    # print(peak_coordinates)

    return median_template, peak_coordinates



def calculate_peak_mask(sorter_path, input_folder, min_distance=5, threshold_abs=None, threshold_rel=0.1, dilation_iterations=3):
    """
    This function creates a median template, detects peaks, removes these peaks from each template, and saves them to a new folder.

    Args:
        input_folder: Path to the original templates (.npy files).
        output_folder: Path for the processed templates.
        min_distance: Minimum distance between peaks.
        threshold_abs: Absolute intensity threshold for peak detection.
        threshold_rel: Relative intensity threshold for peak detection (fraction of max).
        dilation_iterations: Number of iterations for dilating the peak mask.
    """

    # Create paths
    input_path = os.path.join(sorter_path, input_folder)
    
    # Load Templates
    template_files = [f for f in os.listdir(input_path) if f.endswith('.npy')]
    templates = []
    for file in template_files:
        filepath = os.path.join(input_path, file)
        template = np.load(filepath)  # Load using np.load for .npy files
        templates.append(template)

    # Average and Find Peaks (using the skimage function)
    median_template, peak_coordinates = identify_ubiquitous_peaks(templates, 
                                                                  min_distance = min_distance, 
                                                                  threshold_abs = threshold_abs, 
                                                                  threshold_rel = threshold_rel)

    # Create a Mask for Peak Removal
    peak_mask = np.zeros(median_template.shape[:2], dtype=bool)
    for x, y, z in peak_coordinates:
        peak_mask[x, y] = True

    # Dilate the mask, so we also remove the region around the peak, not only the peak itself
    struct = generate_binary_structure(2, 1)
    dilated_peak_mask = binary_dilation(peak_mask, structure=struct, iterations=dilation_iterations)
    
    return dilated_peak_mask, templates, template_files


def peak_mask_processing(sorter_path, output_folder, dilated_peak_mask, templates, template_files, ais_rad=10):
    """
    Processes templates, removing peaks identified by a 2D dilated mask,
    preserving a 2D region around the axon initial segment (AIS).

    Args:
        sorter_path: Base path.
        output_folder: Output folder name.
        dilated_peak_mask: The boolean mask of regions to remove.
        templates: List of numpy arrays.
        template_files: List of filenames.
        ais_rad: Radius around the AIS to protect (2D).
    """
    output_path = os.path.join(sorter_path, output_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    processed_templates = []
    for i, template in enumerate(templates):
        processed_template = template.copy()

        # Find AIS location (maximum value in the template)
        ais_flattened_index = np.argmax(template)
        ais_x, ais_y, ais_z = np.unravel_index(ais_flattened_index, template.shape)

        # Create a mask for AIS protection (circular)
        x, y = np.indices(template.shape[:2])  # Only first two dimensions
        ais_protection_coor = (np.sqrt((x - ais_x)**2 + (y - ais_y)**2) <= ais_rad)
        ais_protection_mask = np.zeros(template.shape[:2], dtype=bool)
        ais_protection_mask[ais_protection_coor] = True

        # Combine the masks:
        removal_mask = dilated_peak_mask & ~ais_protection_mask

        # Apply the removal mask to the template
        processed_template[removal_mask] = 0

        processed_templates.append(processed_template)
        output_filename = os.path.join(output_path, template_files[i])
        np.save(output_filename, processed_template)

    print(f"Processed templates saved to: {output_folder}")
    return processed_templates




# ---------------------------------------------- Extra - Removal of noisy templates ---------------------------------------------------- #





def show_template(template, group, snr, noise):
    temp_diff = np.diff(template)
    tmp_filt = nd.gaussian_filter(temp_diff, sigma=1)
    plt.imshow(np.min(tmp_filt, axis=2).T, vmin=-10, vmax=0, cmap="gist_heat")
    plt.title(f"Shown: {group} - SNR: {snr:.2f} - Noise: {noise:.2f}")
    plt.show()

def remove_noisy_templates(templates, template_files, noise_threshold=0.5, snr_threshold=100, show_which="none"):
    """
    Removes templates with a low signal-to-noise ratio (SNR).

    Args:
        templates: List of 3D numpy arrays.
        noise_threshold: Minimum noise level to check a template for SNR.
        snr_threshold: Minimum SNR to keep a template.
        show_which: Which templates to show ("good", "bad", "all", "none").

    Returns:
        List of templates with Noise below threshold or SNR above the threshold.
    """
    good_templates = templates.copy()
    unique_files = template_files.copy()
    bad_templates = []
    for i, template in enumerate(good_templates):
        noise = np.median(generate_noise_matrix(template, mode="mad"))
        signal = np.abs(np.min(template))
        snr = signal / noise

        # Only for high noise levels, we want to consider removing them
        if noise > noise_threshold:
            
            if snr < snr_threshold:
                bad_templates.append(template)
                good_templates.pop(i)
                unique_files.pop(i)
                
                if show_which == "bad":
                    show_template(template, show_which, snr, noise)
        
        
        # Showing the templates
        if noise < noise_threshold or snr > snr_threshold:
            if show_which == "good":
                show_template(template, show_which, snr, noise)
        
        if show_which == "all": 
            show_template(template, show_which, snr, noise)

    return good_templates, unique_files




# ---------------------------------------------- Extra - Removal of duplicates ---------------------------------------------------- #




def keep_percentile_2D(template, percentile=10):
    """
    Keeps only the most negative signal values in a template,
    setting all other values to zero.

    Args:
        template: A NumPy array (presumably 3D, but works for any dimension).
        Percentile: How many percent should be kept

    Returns:
        Thresholded NumPy array flattened to 2D.
    """
    if not isinstance(template, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    template_2D = np.min(template, axis=2)

    # Identify all negative values
    negative_mask = template_2D < 0
    negative_values = template_2D[negative_mask]

    # Determine the threshold at the corresponding percentile
    threshold_value = np.percentile(negative_values, percentile)

    # Create a mask for the values to keep
    keep_mask = (template_2D <= threshold_value) & negative_mask

    # Use the mask on the output template
    output_template = template_2D.copy()
    output_template[~keep_mask] = 0

    return output_template


def show_duplicates(template1, peak1, i, template2, peak2, j):
    """
    Show the duplicates and their corresponding AIS given the templates and their peaks.
    """
    temp_diff = np.diff(template1)
    tmp_filt = nd.gaussian_filter(temp_diff, sigma=1)
    plt.subplot(1, 2, 1)
    plt.imshow(
        np.min(tmp_filt, axis=2).T, vmin=-10, vmax=0, cmap="gist_heat"
    )
    plt.scatter(peak1[0], peak1[1], marker='x', color='blue', s=20)
    plt.title(f"Template {i}")

    temp_diff2 = np.diff(template2)
    tmp_filt2 = nd.gaussian_filter(temp_diff2, sigma=1)
    plt.subplot(1, 2, 2)
    plt.imshow(
        np.min(tmp_filt2, axis=2).T, vmin=-10, vmax=0, cmap="gist_heat"
    )
    plt.scatter(peak2[0], peak2[1], marker='x', color='blue', s=20)
    plt.title(f"Template {j}")
    plt.show()

    print(f"Peak 1: {peak1} and Peak 2: {peak2}")


def remove_duplicate_templates_euc(templates, filtered_files, params, dist_threshold=5, corr_threshold=0.9, percentile=10):
    """
    Removes duplicate templates by comparing the templates to each other.
    Args:
        processed_templates: List of 3D numpy arrays.
        params: Dictionary with parameters.
        rad: Radius which is considered as the same AIS
        threshold: Correlation threshold for considering two templates as duplicates.
    Returns:
        List of unique templates.
    """

    # Find all the AIS on the templates and remove the ones without an AIS
    peaks = []
    unique_files_temp = filtered_files.copy()
    for i, template in enumerate(templates):
        matrix_capped, peak = localize_ais(template, params)
        if peak.size != 0:
            peaks.append(peak)
        else:
            templates.pop(i)
            unique_files_temp.pop(i)
    

    to_remove = np.array([False for _ in range(len(templates))])
    m = len(peaks)
    euc_dist = pdist(peaks, 'euclidean')
    # condensed matrix: distance saved at m * i + j - ((i + 2) * (i + 1)) // 2
    print(euc_dist)


    for idx, dist in enumerate(euc_dist):
        
        # Check for elements that are below the distance threshold
        if dist < dist_threshold:
            rows, cols = np.triu_indices(m, k=1)
            i = rows[idx]
            j = cols[idx]

            # Flatten everything except the 10 negative percentile and return 2D array
            percentiled_template1 = keep_percentile_2D(templates[i], percentile)
            percentiled_template2 = keep_percentile_2D(templates[j], percentile)

            flattened_template1 = percentiled_template1.flatten()
            flattened_template2 = percentiled_template2.flatten()

            double_zeros = np.array([False for _ in range(flattened_template1.size)])
            for k, val in enumerate(flattened_template1):
                if flattened_template1[k] == 0 and flattened_template2[k] == 0:
                    double_zeros[k] = True
            
            flattened_template1 = flattened_template1[~double_zeros]
            flattened_template2 = flattened_template2[~double_zeros]

            # Only correlate the non-zero values
            correlation = np.corrcoef(flattened_template1, flattened_template2)[0, 1]
            # print(f"Correlation between {i} and {j}: {correlation: .2f}")
            
            # We classify them as duplicates if the correlation is below the threshold
            if correlation > corr_threshold:
                print(f"Template {i} and {j} are duplicates")
                noise_level1 = np.mean(generate_noise_matrix(templates[i], mode="mad"))
                noise_level2 = np.mean(generate_noise_matrix(templates[j], mode="mad"))
                
                if noise_level1 < noise_level2:
                    show_duplicates(templates[i], peaks[i], i, templates[j], peaks[j], j)
                    to_remove[j] = True
                    print(f"Template {j} was removed")
                else:
                    show_duplicates(templates[i], peaks[i], i, templates[j], peaks[j], j)
                    to_remove[i] = True
                    print(f"Template {i} was removed")

    
    unique_templates = []
    unique_files = []
    for i, template in enumerate(templates):
        if ~to_remove[i]:
            unique_templates.append(template)
            unique_files.append(unique_files_temp[i])

    return unique_templates, unique_files