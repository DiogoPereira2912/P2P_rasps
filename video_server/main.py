import streamlit as st
import streamlit.components.v1 as components


def run():
    cols = st.columns(2)
    with cols[0]:
        iframe_src = "http://localhost:8000/video/0"
        components.iframe(iframe_src, height=480)

    with cols[1]:
        iframe_src = "http://localhost:8000/video/1"
        components.iframe(iframe_src, height=480)

    iframe_src = "http://localhost:8000/video/0"
    components.iframe(iframe_src, height=480)


# You can add height and width to the component of course.

if __name__ == "__main__":
    run()
