from analyse import analyse
from process import process
from utils.paths import DTW_Z_V2_CSV, YOUNG_ADULTS_1, YOUNG_ADULTS_2
from utils.read_csv import read_from_dtw_file


def main():
    process(input_files=[YOUNG_ADULTS_1, YOUNG_ADULTS_2])
    distance_df = read_from_dtw_file(DTW_Z_V2_CSV)
    analyse(distance_df)