import pandas as pd
from src import config
from ydata_profiling import ProfileReport
import os



def generate_classifier_report():
    reports_dir = config.reports_eda_dir
    file_name  = config.file_name_reports

    # check root report
    if not os.path.isdir(reports_dir):
        os.makedirs(reports_dir)

    # read data
    df = pd.read_csv(config.data_raw_path)

    # create report
    profile = ProfileReport(df,title=file_name, explorative=True)

    # root file
    report_path = os.path.join(reports_dir,file_name)

    # overwrite
    profile.to_file(report_path)

    print("report file create at: {}".format(report_path))

if __name__ == '__main__':
    generate_classifier_report()