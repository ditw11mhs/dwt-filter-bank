from os import name
import streamlit as st
import numpy as np
from backend_dwt_filter_bank import Solver
import plotly as pl
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd


class Plotter:
    def one_plotter_a(self, data_input):
        fig = make_subplots(
            rows=9,
            cols=1,
            subplot_titles=(
                "ECG",
                "Scale 1",
                "Scale 2",
                "Scale 3",
                "Scale 4",
                "Scale 5",
                "Scale 6",
                "Scale 7",
                "Scale 8",
            ),
        )

        fig.append_trace(
            go.Scattergl(
                # x=data_input["Time ECG"],
                x=np.arange(len(data_input["Data"])),
                y=data_input["Data"],
                mode="lines",
                name=data_input["Data Name"],
            ),
            row=1,
            col=1,
        )
        for scale in np.arange(1, 9):
            fig.append_trace(
                go.Scattergl(
                    # x=data_input["Time ECG"],
                    x=np.arange(len(data_input[f"""C{str(scale)}"""])),
                    y=data_input[f"C{str(scale)}"],
                    mode="lines",
                    name=f"Scale {str(scale)}",
                ),
                row=1 + scale,
                col=1,
            )
        fig.update_layout(height=2000, width=700, title_text="DWT Filter Bank")
        st.plotly_chart(fig, use_container_width=True)

    def one_plotter_b_search(self, data_input):
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=data_input["Time ECG"],
                y=data_input["Data"],
                mode="lines",
                name=data_input["Data Name"],
            ),
        )
        fig.add_trace(
            go.Scattergl(
                x=data_input["Time ECG"],
                y=data_input[f"""C{data_input["T Selection"][1]}"""],
                mode="lines",
                name=f"""C{data_input["T Selection"][1]}""",
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=data_input["Time ECG"], y=data_input["Threshold Out"], line_shape="hv"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    def one_plotter_b_segmentation(self, data_input):
        fig = make_subplots(
            rows=5,
            cols=1,
            subplot_titles=(
                "ECG",
                "QRS Segmentation",
                "T Segmentation",
                "P Segmentation",
                "PQRST Segmentation",
            ),
        )

        fig.append_trace(
            go.Scattergl(
                x=data_input["Time ECG"],
                y=data_input["Data"],
                mode="lines",
                name=data_input["Data Name"],
            ),
            row=1,
            col=1,
        )

        fig.append_trace(
            go.Scattergl(
                x=data_input["Time ECG"],
                y=data_input["QRS"],
                line_shape="hv",
                name="QRS",
            ),
            row=2,
            col=1,
        )

        fig.append_trace(
            go.Scattergl(
                x=data_input["Time ECG"],
                y=data_input["T Wave"],
                line_shape="hv",
                name="T Segmentation",
            ),
            row=3,
            col=1,
        )

        fig.append_trace(
            go.Scattergl(
                x=data_input["Time ECG"],
                y=data_input["P Wave"],
                line_shape="hv",
                name="P Segmentation",
            ),
            row=4,
            col=1,
        )

        fig.append_trace(
            go.Scattergl(
                x=data_input["Time ECG"],
                y=data_input["QRS"],
                line_shape="hv",
                name="QRS",
            ),
            row=5,
            col=1,
        )

        fig.append_trace(
            go.Scattergl(
                x=data_input["Time ECG"],
                y=data_input["T Wave"],
                line_shape="hv",
                name="T Segmentation",
            ),
            row=5,
            col=1,
        )

        fig.append_trace(
            go.Scattergl(
                x=data_input["Time ECG"],
                y=data_input["P Wave"],
                line_shape="hv",
                name="P Segmentation",
            ),
            row=5,
            col=1,
        )

        fig.update_layout(height=1500, width=700)
        st.plotly_chart(fig, use_container_width=True)

    def two_plotter(self, data_input):
        fig = make_subplots(
            rows=5,
            cols=3,
            subplot_titles=(
                "Heel",
                "Heel Frequency",
                "Filtered Heel",
                "Toe",
                "Toe Frequency",
                "Filtered Toe",
                "Hip",
                "Hip Frequency",
                "Filtered Hip",
                "Knee",
                "Knee Frequency",
                "Filtered Knee",
                "Ankle",
                "Ankle Frequency",
                "Filtered Ankle",
            ),
        )
        name_list = ["Heel", "Toe", "Hip", "Knee", "Ankle"]
        
        for i in range(5):
            fig.append_trace(
                go.Scattergl(
                    x=data_input["Time"],
                    y=data_input[name_list[i]],
                    mode="lines",
                    name=name_list[i],
                ),
                row=i+1,
                col=1,
            )

            fig.append_trace(
                go.Scattergl(
                    x=data_input[name_list[i] + " FFT"]["Frequency"],
                    y=data_input[name_list[i] + " FFT"]["Magnitude"],
                    mode="lines",
                    name=name_list[i] + " FFT",
                ),
                row=i+1,
                col=2,
            )
            fig.append_trace(
                go.Scattergl(
                    x=data_input["Time"],
                    y=data_input[name_list[i]],
                    mode="lines",
                    name=name_list[i],
                ),
                row=i+1,
                col=3,
            )
            fig.append_trace(
                go.Scattergl(
                    x=data_input["Time"],
                    y=data_input["Filtered "+name_list[i]]["Forward"],
                    mode="lines",
                    name="Forward "+name_list[i],
                ),
                row=i+1,
                col=3,
            )
            fig.append_trace(
                go.Scattergl(
                    x=data_input["Time"],
                    y=data_input["Filtered "+name_list[i]]["Backward"],
                    mode="lines",
                    name="Backward "+name_list[i],
                ),
                row=i+1,
                col=3,
            )
            
            
            
        fig.update_layout(height=1500, width=700)
        st.plotly_chart(fig, use_container_width=True)


class Main(Solver, Plotter):
    def __init__(self):
        super().__init__()

    def main(self):
        st.title("DWT Filter Bank (Unfinished)")
        self.one_front()
        self.two_front()

    def one_front(self):
        st.markdown(
            """1.  Develop a QRS, P,  and T waves detection based on real-time DWT filter bank with 8 scales as you designed in previous week. Test your program use normal ecg data. Convert the detected data of the waves to Heart Rate, QRS durations, ST segment, PR segment, PR interval, and QT interval"""
        )
        st.markdown("""Current Problem
                    - Thresholding T4 T5 (zero crossing detection done, need to detect if zero crossing is in between peaks found from thresholding)""")
        with st.expander("DWT Filter Bank"):
            c1, c2 = st.columns(2)
            data_name = c1.selectbox("Data to DWT", ["MLII", "V5"])
            split = c2.slider(
                "Split data", min_value=1, max_value=21600, value=(0, 21599)
            )
            dwt_data = self.one_solver_a(data_name, split)
            self.one_plotter_a(dwt_data)
        with st.expander("Segmentation"):
            c1, c2, c3, c4, c5 = st.columns(5)
            c6, c7 = st.columns(2)
            one_solver_b_data = {}
            one_solver_b_data["T1"] = c1.number_input(
                "T1 Delay", min_value=0, max_value=21600, value=1
            )
            one_solver_b_data["T2"] = c2.number_input(
                "T2 Delay", min_value=0, max_value=21600, value=1
            )
            one_solver_b_data["T3"] = c3.number_input(
                "T3 Delay", min_value=0, max_value=21600
            )
            one_solver_b_data["T4"] = c4.number_input(
                "T4 Delay", min_value=0, max_value=21600
            )
            one_solver_b_data["T5"] = c5.number_input(
                "T5 Delay", min_value=0, max_value=21600
            )
            one_solver_b_data["Threshold 1"] = c1.number_input(
                "Threshold 1", min_value=0.0, max_value=10.0, value=0.12
            )
            one_solver_b_data["Threshold 2"] = c2.number_input(
                "Threshold 2", min_value=0.0, max_value=10.0, value=0.17
            )
            one_solver_b_data["Threshold 3"] = c3.number_input(
                "Threshold 3", min_value=0.0, max_value=10.0, value=0.14
            )
            one_solver_b_data["Threshold 4"] = c4.number_input(
                "Threshold 4", min_value=0.0, max_value=10.0
            )
            one_solver_b_data["Threshold 5"] = c5.number_input(
                "Threshold 5", min_value=0.0, max_value=10.0
            )
            one_solver_b_data["Data Name"] = c6.selectbox(
                "Data to Segment", ["MLII", "V5"]
            )
            one_solver_b_data["Split Segment"] = c7.slider(
                "Split Segment", min_value=1, max_value=21600, value=(0, 21599)
            )

            search_mode = st.checkbox("Search Mode")
            if search_mode:
                cl1, cr2 = st.columns(2)
                one_solver_b_data["Search Split"] = cl1.slider(
                    "Search Split Data", min_value=1, max_value=21600, value=(0, 21599)
                )
                one_solver_b_data["T Selection"] = cr2.selectbox(
                    "T Select", ["T1", "T2", "T3", "T4", "T5"]
                )
                search_out = self.one_solver_b_searcher(one_solver_b_data)
                self.one_plotter_b_search(search_out)
            solver_out = self.one_solver_b_segmentation(one_solver_b_data)
            self.one_plotter_b_segmentation(solver_out)

    def two_front(self):
        st.markdown(
            """2. Develop a computer program to design filter digital with zero lag to reduce noise of a kinematical data of gait. Perform a frequency domain analysis to decide filter type. Show effect of filtering in reducing noise and zero leg response.  """
        )
        with st.expander("Filter"):
            two_solver_input = {}

            c1, c2, c3, c4, c5 = st.columns(5)
            two_solver_input["fc Heel"] = c1.number_input(
                "Heel Cutoff Frequency", 0.0, 500.0, 25.0
            )
            two_solver_input["fc Toe"] = c2.number_input(
                "Toe Cutoff Frequency", 0.0, 500.0, 25.0
            )
            two_solver_input["fc Hip"] = c3.number_input(
                "Hip Cutoff Frequency", 0.0, 500.0, 25.0
            )
            two_solver_input["fc Knee"] = c4.number_input(
                "Knee Cutoff Frequency", 0.0, 500.0, 25.0
            )
            two_solver_input["fc Ankle"] = c5.number_input(
                "Ankle Cutoff Frequency", 0.0, 500.0, 25.0
            )
            freq_domain = self.two_solver(two_solver_input)
            self.two_plotter(freq_domain)


if __name__ == "__main__":
    st.set_page_config(page_title="DWT Filter Bank", layout="wide")
    main = Main()
    main.main()
