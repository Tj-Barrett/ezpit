import json
import os

import numpy as np
import pyqtgraph.exporters
from PySide6.QtWidgets import QFileDialog, QMessageBox


# [EN] Shared writer for save_graph_data's save_xy_data/save_extra_data and
#      save_calculated_graph_data's save_xy_data, which were three separate
#      copies of this same validate-then-savetxt logic. Consolidated here;
#      each caller below only supplies its own file-naming convention.
# [KR] save_graph_data의 save_xy_data/save_extra_data와
#      save_calculated_graph_data의 save_xy_data가 각각 이 검증+저장 로직을
#      복사해 갖고 있었습니다. 여기로 통합했고, 아래 호출부는 각자의 파일
#      이름 규칙만 정의합니다.
def _save_xy(full_path, x, y, file_label):
    if x is None or y is None or len(x) != len(y) or len(x) == 0:
        return
    np.savetxt(os.path.join(full_path, file_label), np.column_stack([x, y]), comments="")


def save_selected_graphs(parent, checkboxes, plots, save_dir, base_name="plot"):
    for i, (cb, plot) in enumerate(zip(checkboxes, plots, strict=False)):
        if cb.isChecked():
            file_name = f"{base_name}_{i}.png"
            file_path = os.path.join(save_dir, file_name)

            exporter = pyqtgraph.exporters.ImageExporter(plot.plotItem)
            exporter.export(file_path)


def save_graph_data(
    xs,
    ys,
    raw_iq=None,
    background_data=None,
    sq_original_data=None,
    sq_polynomial_data=None,
    fq_smoothed_data=None,
    gr=None,
    r=None,
    mean_sq_fi=None,
    sq_mean_fi=None,
    r_smoothed_data=None,
    G_smoothed_data=None,
    folder_name="output",
    path=".",
    extensions_to_save=None,
):
    full_path = os.path.join(path, folder_name)
    os.makedirs(full_path, exist_ok=True)

    if extensions_to_save is None:
        extensions_to_save = [
            ".iq",
            ".sq",
            ".fq",
            ".gr",
            ".mean_sq_fi.npy",
            ".sq_mean_fi.npy",
            ".gr_smoothed",
        ]

    # [EN] Main data always saves as "{folder_name}{extension}" — no suffix.
    # [KR] 메인 데이터는 항상 "{폴더명}{확장자}" 형식으로 저장됩니다 (suffix 없음).
    def save_xy_data(x, y, extension):
        _save_xy(full_path, x, y, f"{folder_name}{extension}")

    if xs and ys:
        if ".iq" in extensions_to_save and len(xs) > 0 and len(ys) > 0:
            save_xy_data(xs[0], ys[0], ".iq")
        if ".sq" in extensions_to_save and len(xs) > 1 and len(ys) > 1:
            save_xy_data(xs[1], ys[1], ".sq")
        if ".fq" in extensions_to_save and len(xs) > 2 and len(ys) > 2:
            save_xy_data(xs[2], ys[2], ".fq")
        if ".gr" in extensions_to_save:
            x_gr = r if r is not None else (xs[3] if len(xs) > 3 else None)
            y_gr = gr if gr is not None else (ys[3] if len(ys) > 3 else None)
            save_xy_data(x_gr, y_gr, ".gr")

    # [EN] Secondary/extra data (raw, background, original, smoothed, ...) needs
    #      a suffix so it doesn't overwrite the main file written above.
    # [KR] 부가 데이터(raw, background, original, smoothed 등)는 위에서 저장한
    #      메인 파일을 덮어쓰지 않도록 suffix가 필요합니다.
    def save_extra_data(x, y, extension, suffix):
        _save_xy(full_path, x, y, f"{folder_name}_{suffix}{extension}")

    if ".iq" in extensions_to_save and len(xs) > 0 and xs[0] is not None:
        if raw_iq is not None and raw_iq[0] is not None and raw_iq[1] is not None:
            save_extra_data(raw_iq[0], raw_iq[1], ".iq", "raw")
        if background_data is not None and background_data[0] is not None and background_data[1] is not None:
            save_extra_data(background_data[0], background_data[1], ".iq", "background")

    if ".sq" in extensions_to_save and len(xs) > 1 and xs[1] is not None:
        if sq_original_data is not None:
            save_extra_data(xs[1], sq_original_data, ".sq", "original")
        if sq_polynomial_data is not None:
            save_extra_data(xs[1], sq_polynomial_data, ".sq", "polynomial")

    if ".fq" in extensions_to_save and len(xs) > 2 and xs[2] is not None:
        if fq_smoothed_data is not None:
            save_extra_data(xs[2], fq_smoothed_data, ".fq", "smoothed")

    if ".gr" in extensions_to_save or ".gr_smoothed" in extensions_to_save:
        if r_smoothed_data is not None and G_smoothed_data is not None:
            save_extra_data(r_smoothed_data, G_smoothed_data, ".gr", "smoothed")

    if len(xs) > 0 and xs[0] is not None:
        q = xs[0]
        if mean_sq_fi is not None:
            mean_array = np.full_like(q, mean_sq_fi) if not isinstance(mean_sq_fi, np.ndarray) else mean_sq_fi
            if ".mean_sq_fi.npy" in extensions_to_save:
                out_path = os.path.join(full_path, f"{folder_name}.mean_sq_fi.npy")
                np.save(out_path, mean_array)

        if sq_mean_fi is not None:
            sqmean_array = np.full_like(q, sq_mean_fi) if not isinstance(sq_mean_fi, np.ndarray) else sq_mean_fi
            if ".sq_mean_fi.npy" in extensions_to_save:
                out_path = os.path.join(full_path, f"{folder_name}.sq_mean_fi.npy")
                np.save(out_path, sqmean_array)


def save_calculated_graph_data(xs, ys, gr=None, r=None, folder_name="output_cal", path=".", extensions_to_save=None):
    full_path = os.path.join(path, folder_name)
    os.makedirs(full_path, exist_ok=True)

    if extensions_to_save is None:
        extensions_to_save = [".caliq", ".calsq", ".calfq", ".calgr"]

    def save_xy_data(x, y, extension):
        _save_xy(full_path, x, y, f"{folder_name}{extension}")

    if xs and ys:
        if ".caliq" in extensions_to_save and len(xs) > 0:
            save_xy_data(xs[0], ys[0], ".caliq")
        if ".calsq" in extensions_to_save and len(xs) > 1:
            save_xy_data(xs[1], ys[1], ".calsq")
        if ".calfq" in extensions_to_save and len(xs) > 2:
            save_xy_data(xs[2], ys[2], ".calfq")
        if ".calgr" in extensions_to_save:
            x_gr = r if r is not None else (xs[3] if len(xs) > 3 else None)
            y_gr = gr if gr is not None else (ys[3] if len(ys) > 3 else None)
            save_xy_data(x_gr, y_gr, ".calgr")


def save_compton_data(q, intensity, output_dir, file_prefix, fmt="%.6f"):
    q = np.asarray(q)
    intensity = np.asarray(intensity)

    if len(q) != len(intensity):
        raise ValueError("Length of q and intensity must match.")

    final_filename = os.path.splitext(file_prefix)[0] + ".compton"
    filepath = os.path.join(output_dir, final_filename)

    header = "Q (A-1)\tCompton Scattering Intensity"
    data = np.column_stack([q, intensity])

    try:
        np.savetxt(filepath, data, fmt=fmt, header=header, comments="# ")
        print(f"Compton data successfully saved to: {filepath}")
    except Exception as e:
        print(f"Error saving Compton data to {filepath}: {e}")


def write_config_file(config, path):
    with open(path, "w") as f:
        json.dump(config, f, indent=4)


def write_compton_file(x, y, parent=None):
    file_path, _ = QFileDialog.getSaveFileName(parent, "Save Compton File", filter="Compton Files (*.compton)")

    if not file_path:
        return

    if not file_path.endswith(".compton"):
        file_path += ".compton"

    try:
        data = np.column_stack([x, y])
        np.savetxt(file_path, data, fmt="%.6f", comments="", header="Q Compton")

        QMessageBox.information(
            parent,
            "Save Successful",
            f"Compton file saved successfully to:\n{file_path}",
        )

    except Exception as e:
        QMessageBox.critical(parent, "Save Error", f"Failed to save Compton file:\n{e}")
