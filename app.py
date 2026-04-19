import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from corner_model import CornerPredictionModel, MatchDataParser

# ======================================================================
# KONFIGURÁCIÓ
# ======================================================================

st.set_page_config(
    page_title="Corner Prediction v13.3",
    page_icon="⚽",
    layout="wide",
)
