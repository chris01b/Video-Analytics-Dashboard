import streamlit as st
from deep_list import detect

def main():
    st.title("YOLOv11 Object Detection Dashboard")
    inference_msg = st.empty()
    st.sidebar.title("Configuration")

    input_source = st.sidebar.radio(
        "Select input source",
        ('RTSP', 'Webcam', 'Local video')
    )

    conf_thres = st.sidebar.number_input(
        "Class confidence threshold",
        min_value=0.0, max_value=1.0,
        value=0.25, step=0.01
    )

    conf_thres_drift = st.sidebar.number_input(
        "Class confidence threshold for drift detection",
        min_value=0.0, max_value=1.0,
        value=0.75, step=0.01
    )

    fps_drop_warn_thresh = st.sidebar.number_input(
        "FPS drop warning threshold",
        min_value=1.0, max_value=60.0,
        value=8.0, step=0.5
    )

    # Separate control for saving output video
    save_output_video = st.sidebar.radio("Save output video?", ('Yes', 'No'))
    nosave = False if save_output_video == 'Yes' else True

    display_labels_option = st.sidebar.checkbox("Display Labels", value=True)

    save_poor_frame = st.sidebar.radio("Save poor performing frames?", ('Yes', 'No'))
    save_poor_frame__ = True if save_poor_frame == "Yes" else False

    # New parameter: Display Interval
    display_interval = st.sidebar.number_input(
        "Frame Display Interval",
        min_value=1, max_value=100,  # Adjust as needed
        value=1, step=1,
        help="Display every Nth frame. Default is 1 (every frame)."
    )

    # ------------------------- LOCAL VIDEO ------------------------------
    if input_source == "Local video":
        video = st.sidebar.file_uploader(
            "Select input video",
            type=["mp4", "avi"],
            accept_multiple_files=False
        )

        if st.sidebar.button("Start tracking") and video is not None:
            stframe = st.empty()

            st.subheader("Inference Stats")
            kpi1, kpi2, kpi3 = st.columns(3)

            st.subheader("System Stats")
            js1, js2, js3 = st.columns(3)

            # Updating Inference results
            with kpi1:
                st.markdown("**Frame Rate**")
                kpi1_text = st.markdown("0 FPS")
                fps_warn = st.empty()

            with kpi2:
                st.markdown("**Detected Objects**")
                kpi2_text = st.markdown("0")

            with kpi3:
                st.markdown("**Total Detected Classes**")
                kpi3_text = st.markdown("0")

            # Updating System stats
            with js1:
                st.markdown("**Memory Usage**")
                js1_text = st.markdown("0%")

            with js2:
                st.markdown("**CPU Usage**")
                js2_text = st.markdown("0%")

            with js3:
                st.markdown("**GPU Memory Usage**")
                js3_text = st.markdown("0 MB")

            st.subheader("Inference Overview")
            inf_ov_1, inf_ov_2, inf_ov_3, inf_ov_4 = st.columns(4)

            with inf_ov_1:
                st.markdown(f"**Poor Performing Classes (Conf < {conf_thres_drift})**")
                inf_ov_1_text = st.markdown("None")

            with inf_ov_2:
                st.markdown("**Number of Poor Performing Frames**")
                inf_ov_2_text = st.markdown("0")

            with inf_ov_3:
                st.markdown("**Minimum FPS**")
                inf_ov_3_text = st.markdown("0 FPS")

            with inf_ov_4:
                st.markdown("**Maximum FPS**")
                inf_ov_4_text = st.markdown("0 FPS")

            # Start detection
            detect(
                source=video.name,
                stframe=stframe,
                kpi1_text=kpi1_text,
                kpi2_text=kpi2_text,
                kpi3_text=kpi3_text,
                js1_text=js1_text,
                js2_text=js2_text,
                js3_text=js3_text,
                conf_thres=conf_thres,
                nosave=nosave,
                display_labels=display_labels_option,
                conf_thres_drift=conf_thres_drift,
                save_poor_frame__=save_poor_frame__,
                inf_ov_1_text=inf_ov_1_text,
                inf_ov_2_text=inf_ov_2_text,
                inf_ov_3_text=inf_ov_3_text,
                inf_ov_4_text=inf_ov_4_text,
                fps_warn=fps_warn,
                fps_drop_warn_thresh=fps_drop_warn_thresh,
                display_interval=display_interval
            )

            inference_msg.success("Inference Complete!")

    # -------------------------- WEBCAM ----------------------------------
    if input_source == "Webcam":
        if st.sidebar.button("Start tracking"):
            stframe = st.empty()
            st.subheader("Inference Stats")
            kpi1, kpi2, kpi3 = st.columns(3)

            st.subheader("System Stats")
            js1, js2, js3 = st.columns(3)

            # Updating Inference results
            with kpi1:
                st.markdown("**Frame Rate**")
                kpi1_text = st.markdown("0 FPS")
                fps_warn = st.empty()

            with kpi2:
                st.markdown("**Detected Objects**")
                kpi2_text = st.markdown("0")

            with kpi3:
                st.markdown("**Total Detected Classes**")
                kpi3_text = st.markdown("0")

            # Updating System stats
            with js1:
                st.markdown("**Memory Usage**")
                js1_text = st.markdown("0%")

            with js2:
                st.markdown("**CPU Usage**")
                js2_text = st.markdown("0%")

            with js3:
                st.markdown("**GPU Memory Usage**")
                js3_text = st.markdown("0 MB")

            st.subheader("Inference Overview")
            inf_ov_1, inf_ov_2, inf_ov_3, inf_ov_4 = st.columns(4)

            with inf_ov_1:
                st.markdown(f"**Poor Performing Classes (Conf < {conf_thres_drift})**")
                inf_ov_1_text = st.markdown("None")

            with inf_ov_2:
                st.markdown("**Number of Poor Performing Frames**")
                inf_ov_2_text = st.markdown("0")

            with inf_ov_3:
                st.markdown("**Minimum FPS**")
                inf_ov_3_text = st.markdown("0 FPS")

            with inf_ov_4:
                st.markdown("**Maximum FPS**")
                inf_ov_4_text = st.markdown("0 FPS")

            # Start detection
            detect(
                source='0',
                stframe=stframe,
                kpi1_text=kpi1_text,
                kpi2_text=kpi2_text,
                kpi3_text=kpi3_text,
                js1_text=js1_text,
                js2_text=js2_text,
                js3_text=js3_text,
                conf_thres=conf_thres,
                nosave=nosave,
                display_labels=display_labels_option,
                conf_thres_drift=conf_thres_drift,
                save_poor_frame__=save_poor_frame__,
                inf_ov_1_text=inf_ov_1_text,
                inf_ov_2_text=inf_ov_2_text,
                inf_ov_3_text=inf_ov_3_text,
                inf_ov_4_text=inf_ov_4_text,
                fps_warn=fps_warn,
                fps_drop_warn_thresh=fps_drop_warn_thresh,
                display_interval=display_interval
            )

            inference_msg.success("Inference Complete!")

    # -------------------------- RTSP ------------------------------------
    if input_source == "RTSP":
        rtsp_input = st.sidebar.text_input(
            "RTSP Stream URL",
            "rtsp://192.168.0.1/stream"
        )

        if st.sidebar.button("Start tracking") and rtsp_input:
            stframe = st.empty()

            st.subheader("Inference Stats")
            kpi1, kpi2, kpi3 = st.columns(3)

            st.subheader("System Stats")
            js1, js2, js3 = st.columns(3)

            # Updating Inference results
            with kpi1:
                st.markdown("**Frame Rate**")
                kpi1_text = st.markdown("0 FPS")
                fps_warn = st.empty()

            with kpi2:
                st.markdown("**Detected Objects**")
                kpi2_text = st.markdown("0")

            with kpi3:
                st.markdown("**Total Detected Classes**")
                kpi3_text = st.markdown("0")

            # Updating System stats
            with js1:
                st.markdown("**Memory Usage**")
                js1_text = st.markdown("0%")

            with js2:
                st.markdown("**CPU Usage**")
                js2_text = st.markdown("0%")

            with js3:
                st.markdown("**GPU Memory Usage**")
                js3_text = st.markdown("0 MB")

            st.subheader("Inference Overview")
            inf_ov_1, inf_ov_2, inf_ov_3, inf_ov_4 = st.columns(4)

            with inf_ov_1:
                st.markdown(f"**Poor Performing Classes (Conf < {conf_thres_drift})**")
                inf_ov_1_text = st.markdown("None")

            with inf_ov_2:
                st.markdown("**Number of Poor Performing Frames**")
                inf_ov_2_text = st.markdown("0")

            with inf_ov_3:
                st.markdown("**Minimum FPS**")
                inf_ov_3_text = st.markdown("0 FPS")

            with inf_ov_4:
                st.markdown("**Maximum FPS**")
                inf_ov_4_text = st.markdown("0 FPS")

            # Start detection
            detect(
                source=rtsp_input,
                stframe=stframe,
                kpi1_text=kpi1_text,
                kpi2_text=kpi2_text,
                kpi3_text=kpi3_text,
                js1_text=js1_text,
                js2_text=js2_text,
                js3_text=js3_text,
                conf_thres=conf_thres,
                nosave=nosave,
                display_labels=display_labels_option,
                conf_thres_drift=conf_thres_drift,
                save_poor_frame__=save_poor_frame__,
                inf_ov_1_text=inf_ov_1_text,
                inf_ov_2_text=inf_ov_2_text,
                inf_ov_3_text=inf_ov_3_text,
                inf_ov_4_text=inf_ov_4_text,
                fps_warn=fps_warn,
                fps_drop_warn_thresh=fps_drop_warn_thresh,
                display_interval=display_interval
            )

            inference_msg.success("Inference Complete!")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
