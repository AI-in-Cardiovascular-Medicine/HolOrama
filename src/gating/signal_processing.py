import time
import numpy as np
from loguru import logger
from scipy.signal import find_peaks, butter, filtfilt
import cv2


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    return wrapper


@timing_decorator
def prepare_data(main_window, frames, report_data, x1=50, x2=450, y1=50, y2=450):
    """Prepares data for plotting."""
    try:
        gating_signal = main_window.gating_signal
        if not gating_signal:  # skip if empty
            raise KeyError
        if gating_signal['gating_config'] == main_window.config.gating and len(
            gating_signal['image_based_gating']
        ) == len(frames):
            return (
                gating_signal['image_based_gating'],
                gating_signal['contour_based_gating'],
                gating_signal['image_based_gating_filtered'],
                gating_signal['contour_based_gating_filtered'],
            )
    except KeyError:
        pass

    # Initialize variables
    step = main_window.config.gating.normalize_step
    maxima_only = main_window.config.gating.maxima_only
    # Crop frames to a specific region
    frames = frames[:, x1:x2, y1:y2]

    # Normalize signals
    correlation = normalize_data(calculate_correlation(frames), step)
    blurring = normalize_data(calculate_blurring_fft(frames), step)
    # shortest_dist = normalize_data(report_data['shortest_distance'], step)
    # vector_angle = normalize_data(report_data['vector_angle'], step)
    # vector_length = normalize_data(report_data['vector_length'], step)
    # Shift contour signals to align with current frame
    shortest_dist = normalize_data(np.roll(report_data['shortest_distance'], 1), step)
    vector_angle = normalize_data(np.roll(report_data['vector_angle'], 1), step)
    vector_length = normalize_data(np.roll(report_data['vector_length'], 1), step)
    
    # Set first frame to 0 (no previous frame)
    shortest_dist[0] = 0
    vector_angle[0] = 0
    vector_length[0] = 0

    signal_image_based_filtered = [
        bandpass_filter(main_window, correlation),
        bandpass_filter(main_window, blurring),
    ]
    signal_contour_based_filtered = [
        bandpass_filter(main_window, shortest_dist),
        bandpass_filter(main_window, vector_angle),
        bandpass_filter(main_window, vector_length),
    ]
    image_based_gating = combined_signal(main_window, [correlation, blurring], maxima_only=maxima_only)
    image_based_gating_filtered = combined_signal(main_window, signal_image_based_filtered, maxima_only=maxima_only)
    contour_based_gating = combined_signal(main_window, [shortest_dist, vector_angle, vector_length], maxima_only=False)
    contour_based_gating_filtered = combined_signal(main_window, signal_contour_based_filtered, maxima_only=False)

    main_window.gating_signal = {
        'image_based_gating': list(image_based_gating),
        'contour_based_gating': list(contour_based_gating),
        'image_based_gating_filtered': list(image_based_gating_filtered),
        'contour_based_gating_filtered': list(contour_based_gating_filtered),
        'gating_config': dict(main_window.config.gating),
    }

    return image_based_gating, contour_based_gating, image_based_gating_filtered, contour_based_gating_filtered


def normalize_data(data, step):
    # z-score normalization either for full set, or defined steps
    if step == 0:
        return (data - np.mean(data)) / np.std(data)
    else:
        normalized_data = np.zeros_like(data)

        for i in range(0, len(data), step):
            segment = data[i : i + step]

            segment_normalized = (segment - np.mean(segment)) / np.std(segment)

            normalized_data[i : i + step] = segment_normalized

        return normalized_data


@timing_decorator
def calculate_correlation(frames):
    """Calculates correlation coefficients between consecutive frames."""
    correlations = []
    for i in range(len(frames) - 1):
        corr = np.corrcoef(frames[i].ravel(), frames[i + 1].ravel())[0, 1]
        correlations.append(corr)
    correlations.append(0)  # to match the length of the frames

    return correlations


@timing_decorator
def calculate_blurring_fft(frames):
    """Calculates blurring using Fast Fourier Transform. Takes the average of the 10% highest frequencies."""
    blurring_scores = []
    for frame in frames:
        fft_data = np.fft.fft2(frame)
        fft_shifted = np.fft.fftshift(fft_data)
        magnitude_spectrum = np.abs(fft_shifted)

        # Use np.partition to get the 10% highest frequencies
        n = len(magnitude_spectrum.ravel())
        threshold_index = int(0.9 * n)
        highest_frequencies = np.partition(magnitude_spectrum.ravel(), threshold_index)[threshold_index:]
        blurring_score = np.mean(highest_frequencies)
        blurring_scores.append(blurring_score)

    return blurring_scores
    # blurring_scores = []
    # for frame in frames:
    #     # use cv2.Laplacian to calculate the blurring, should return same format as above
    #     laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    #     magnitude_spectrum = np.abs(laplacian)

    #     n = len(magnitude_spectrum.ravel())
    #     threshold_index = int(0.9 * n)
    #     highest_frequencies = np.partition(laplacian.ravel(), threshold_index)[threshold_index:]
    #     blurring_score = np.mean(highest_frequencies)
    #     blurring_scores.append(blurring_score)

    # return blurring_scores


def bandpass_filter(main_window, signal):
    """
    Applies a Butterworth bandpass filter to the input signal using instance parameters.

    Parameters:
    - signal (array-like): The input signal to filter.

    Returns:
    - filtered_signal (numpy.ndarray): The bandpass filtered signal.
    """
    lowcut = main_window.config.gating.lowcut
    highcut = main_window.config.gating.highcut
    order = main_window.config.gating.order
    fs = main_window.metadata['frame_rate']  # for Butterworth filter

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')

    # Apply filter using filtfilt for zero phase distortion
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


def combined_signal(
    main_window,
    signal_list,
    maxima_only=False,
):
    """
    Combines multiple signals into one by weighting them based on the variability of their extrema.
    Assumption: The more variable the extrema, the less reliable the signal, since heart rate is regular.

    Parameters:
    - signal_list (list): A list of signals to combine.
    - maxima_only (bool): If True, only maxima are considered for variability calculation.

    Returns:
    - combined_signal (numpy.ndarray): The combined signal.
    """
    # find extrema indices for all curves
    extrema_indices = []
    for signal in signal_list:
        if maxima_only:
            extrema_indices.append(identify_extrema(main_window, signal)[1][::2])
        else:
            extrema_indices.append(identify_extrema(main_window, signal)[0][::2])

    # find variability in extrema indices, based on assumption that heartrate is regular
    variability = []
    for extrema in extrema_indices:
        variability.append(np.std(np.diff(extrema)))

    # calculate sum of all variabilities and then create a combined signal with weights as percent of variability
    # sum_variability = np.sum(variability)
    # weights = [(var / sum_variability) ** -1 for var in variability]
    inverse_variability = [(1 / var) for var in variability]
    weights = [inv_var / sum(inverse_variability) for inv_var in inverse_variability]


    # print the chosen weights per variable
    if len(signal_list) == 3:
        logger.info(f"Signal weights: Shortest distance: {weights[0]:.2f}, Vector angle: {weights[1]:.2f}, Vector length: {weights[2]:.2f}")
    elif len(signal_list) == 2:
        logger.info(f"Signal weights: Correlation: {weights[0]:.2f}, Blurring: {weights[1]:.2f}")

    combined_signal = np.zeros(len(signal_list[0]))
    for i, signal in enumerate(signal_list):
        combined_signal += weights[i] * signal

    return combined_signal


def identify_extrema(main_window, signal):
    extrema_y_lim = main_window.config.gating.extrema_y_lim
    extrema_x_lim = main_window.config.gating.extrema_x_lim

    # Remove NaN and infinite values from the signal
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Dynamically calculate prominence based on the signal's characteristics
    min_height = np.percentile(signal, extrema_y_lim)  # Only consider peaks above the median

    # Find maxima and minima using find_peaks with dynamic prominence
    maxima_indices, _ = find_peaks(signal, distance=extrema_x_lim, height=min_height)
    minima_indices, _ = find_peaks(-signal, distance=extrema_x_lim, height=-min_height)

    # Combine maxima and minima indices into one array and sort them
    extrema_indices = np.concatenate((maxima_indices, minima_indices))
    extrema_indices = np.sort(extrema_indices)

    return extrema_indices, maxima_indices
