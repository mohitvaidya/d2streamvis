import streamlit as st
import base64
import io
from io import BytesIO, StringIO
import odvideo
from odvideo import visual_od
from pull_blob import pull_main
from ffmpeg_main import pre_process
from pushjson import push_files
from oddf import odasdf
import json
import os
import numpy as np
import cv2
import sys
from functools import lru_cache
from time import time


# sys.path = ['', '/home/netramum/anaconda3/envs/visod/lib/python36.zip', '/home/netramum/anaconda3/envs/visod/lib/python3.6', '/home/netramum/anaconda3/envs/visod/lib/python3.6/lib-dynload', '/home/netramum/anaconda3/envs/visod/lib/python3.6/site-packages', '/home/netramum/.local/lib/python3.6/site-packages',  '/data1/code_base/mnt_data/visd2/d2sourcecode/detectron2']

@lru_cache(maxsize=None)
def st_model():
    return odvideo.load_model()


basepath = '/app'
STYLE = """
<style>
img {

    max-width: 50%;
}

</style>
"""

def main_process(basepath = None, video_id=None):
    st.subheader("Duration for the trimmed video")

    duration = st.slider("",1,60,10) 

    st.subheader("FPS for the trimmed video")

    fps = st.slider("fps for the trimmed video",1,25,1) 


    if st.button(label = "Process video",key=0):
        try:
            os.system(f'rm {basepath}/*webm')
            os.system(f'rm {basepath}/*json')
            with st.spinner("processing....."):
                pull_main(video_id=video_id)
                # ffmpeg_id = f'{video_id}_'
                pre_process(video_id = video_id, trim_duration=duration, fps=fps)

            with open(f'{basepath}/{video_id}_.mp4','rb') as f:
                vid = f.read()

            st.video(vid)
            st.success("successfully processed")
        except Exception as e:
            st.exception(f'error during processing video {e}')

    if st.button(label = "Start OD", key=1):
        try:
            with st.spinner("doing inference....."):
                visual_od(video_id = f'{video_id}_', model=st_model())
                st.success("Successfully processed")

        except Exception as e:
            st.exception(f'Got error during OD on video {e}')


    if st.button(label = "Display video output"):
        with open(f'{basepath}/{video_id}_out.webm','rb') as f:
            vid_out = f.read()
        st.subheader("Video output")
        st.video(vid_out)

    if st.button(label = "Display dataframe output", key=5):
        try:
            df = odasdf(video_id)
            "", st.dataframe(df)
            b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
            st.markdown(href, unsafe_allow_html=True)


        except Exception as e:
            st.exception(f'error during writing of JSON file {e}')



def clean_cache():
    with st.spinner("Cleaning....."):
        os.system(f'rm {basepath}/*mp4')
        os.system(f'rm {basepath}/*webm')
        os.system(f'rm {basepath}/*json')


st.set_option('deprecation.showfileUploaderEncoding', False)
st.markdown(STYLE, unsafe_allow_html=True)

menu = ["BLOB Storage", "System Upload"]
choice = st.sidebar.selectbox("Menu", menu)


if choice=="BLOB Storage":

    if st.button(label = "Clean cache"):
        clean_cache()
    st.subheader("Please provide video id from athena production storage account")
    video_id = st.text_input("","15341")

    main_process(basepath= basepath, video_id=video_id)

    if st.button(label = "Display JSON output", key=2):
        try:
            with open(f'{basepath}/{video_id}_.json','r') as f:
                res = json.load(f)
            st.json(res)
        except Exception as e:
            st.exception(f'error during writing of JSON file {e}')


    if st.button(label = "Upload JSON output", key=3):
        try:
            push_files(file=video_id)
            st.success("Successfully uploaded JSON")
        except Exception as e:
            st.exception(f'error during writing of JSON file {e}')

##############################################################################################################################    
if choice=="System Upload":

    if st.button(label = "Clean cache"):
        clean_cache()

    file = st.file_uploader("Upload file", type=["mp4"])
    show_f = st.empty()
    if not file:
        pass
        # show_f.info("Upload file")

    else:
        if isinstance(file, BytesIO):
            show_f.video(file)
            with open(f'{basepath}/file_upload.mp4','wb') as f:
                f.write(file.read())

            video_id = 'file_upload'
            main_process(basepath=basepath, video_id= video_id)


