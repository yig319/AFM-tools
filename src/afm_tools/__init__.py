from afm_tools import afm_image_analyzer
from afm_tools import afm_utils
from afm_tools import afm_viz
from afm_tools import cmap
from afm_tools import domain_analysis
from afm_tools import makevideo

from afm_tools.afm_image_analyzer import (afm_RMS_roughness, afm_analyzer,
                                          afm_line_profiler, bresenham_line,
                                          calculate_height_profile,
                                          detect_dark_holes_local_diff, fft2d,
                                          fill_holes_with_mean, fit_background,
                                          map_roughness, polyfit2d,
                                          remove_surface_particles,)
from afm_tools.afm_utils import (convert_scan_setting, convert_with_unit,
                                 define_percentage_threshold, flexible_round,
                                 format_func, load_waves, parse_ibw,
                                 show_image_stats,)
from afm_tools.afm_viz import (AFMVisualizer, df_scatter, show_peaks,
                               show_pfm_images, tip_potisition_analyzer,
                               violinplot_roughness,)
from afm_tools.cmap import (ScientificColourMaps, define_white_viridis,)
from afm_tools.domain_analysis import (ICA_analysis, convert_binary,
                                       crop_image, domain_rule,
                                       evaluate_ICA_n_components,
                                       find_histogram_peaks, shift_peak,
                                       shift_peak_batch, shift_phase,
                                       show_images, show_interactive_image,)
from afm_tools.makevideo import (VideoMaker, generate_z_voltages,)

_HAS_DRAWING_3D = True
try:
    from afm_tools import drawing_3d
    from afm_tools.drawing_3d import (draw_box, draw_surface, sphere_to_surface,)
except Exception:  # pragma: no cover - optional dependency
    _HAS_DRAWING_3D = False
    drawing_3d = None

__all__ = ['AFMVisualizer', 'ICA_analysis', 'ScientificColourMaps',
           'VideoMaker', 'afm_RMS_roughness', 'afm_analyzer',
           'afm_image_analyzer', 'afm_line_profiler', 'afm_utils', 'afm_viz',
           'bresenham_line', 'calculate_height_profile', 'cmap',
           'convert_binary', 'convert_scan_setting', 'convert_with_unit',
           'crop_image', 'define_percentage_threshold', 'define_white_viridis',
           'detect_dark_holes_local_diff', 'df_scatter', 'domain_analysis',
           'domain_rule',
           'evaluate_ICA_n_components', 'fft2d', 'fib', 'fill_holes_with_mean',
           'find_histogram_peaks', 'fit_background', 'flexible_round',
           'format_func', 'generate_z_voltages', 'load_waves', 'main',
           'makevideo', 'map_roughness', 'parse_args', 'parse_ibw',
           'polyfit2d', 'remove_surface_particles', 'run', 'setup_logging',
           'shift_peak', 'shift_peak_batch', 'shift_phase', 'show_image_stats',
           'show_images', 'show_interactive_image', 'show_peaks',
           'show_pfm_images', 'skeleton',
           'tip_potisition_analyzer', 'violinplot_roughness']

if _HAS_DRAWING_3D:
    __all__.extend(['draw_box', 'draw_surface', 'drawing_3d', 'sphere_to_surface'])
